#include "tsla_lob/csv_reader.hpp"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <limits>
#include <sstream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <utility>

namespace tsla_lob {
namespace {

constexpr char kEmptyInput[] = "";

}  // namespace

MappedFile::MappedFile(const std::string& path) {
  descriptor_ = ::open(path.c_str(), O_RDONLY);
  if (descriptor_ < 0) {
    throw std::runtime_error("cannot open " + path + ": " + std::strerror(errno));
  }

  struct stat metadata {};
  if (::fstat(descriptor_, &metadata) != 0) {
    const std::string reason = std::strerror(errno);
    close();
    throw std::runtime_error("cannot stat " + path + ": " + reason);
  }
  if (metadata.st_size < 0) {
    close();
    throw std::runtime_error("negative file size for " + path);
  }

  size_ = static_cast<std::size_t>(metadata.st_size);
  if (size_ == 0) {
    return;
  }

  void* mapping = ::mmap(nullptr, size_, PROT_READ, MAP_PRIVATE, descriptor_, 0);
  if (mapping == MAP_FAILED) {
    const std::string reason = std::strerror(errno);
    data_ = nullptr;
    close();
    throw std::runtime_error("cannot map " + path + ": " + reason);
  }
  data_ = static_cast<const char*>(mapping);
}

MappedFile::~MappedFile() {
  close();
}

MappedFile::MappedFile(MappedFile&& other) noexcept
    : descriptor_(std::exchange(other.descriptor_, -1)),
      data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)) {}

MappedFile& MappedFile::operator=(MappedFile&& other) noexcept {
  if (this != &other) {
    close();
    descriptor_ = std::exchange(other.descriptor_, -1);
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
  }
  return *this;
}

std::string_view MappedFile::view() const noexcept {
  return size_ == 0 ? std::string_view{} : std::string_view(data_, size_);
}

std::uint64_t MappedFile::size() const noexcept {
  return static_cast<std::uint64_t>(size_);
}

void MappedFile::close() noexcept {
  if (data_ != nullptr) {
    ::munmap(const_cast<char*>(data_), size_);
    data_ = nullptr;
  }
  if (descriptor_ >= 0) {
    ::close(descriptor_);
    descriptor_ = -1;
  }
  size_ = 0;
}

CsvCursor::CsvCursor(std::string_view input, std::string source)
    : begin_(input.empty() ? kEmptyInput : input.data()),
      cursor_(begin_),
      end_(begin_ + input.size()),
      source_(std::move(source)) {}

bool CsvCursor::empty() const noexcept {
  return cursor_ == end_;
}

std::uint64_t CsvCursor::line() const noexcept {
  return line_;
}

Message CsvCursor::read_message() {
  if (empty()) {
    fail("unexpected end of file");
  }

  Message message;
  message.timestamp_ns = read_timestamp_ns();
  consume_comma();
  const std::int64_t event_type = read_integer();
  consume_comma();
  message.order_id = read_integer();
  consume_comma();
  const std::int64_t size = read_integer();
  consume_comma();
  const std::int64_t price = read_integer();
  consume_comma();
  const std::int64_t direction = read_integer();

  if (event_type < static_cast<std::int64_t>(EventType::submission) ||
      event_type > static_cast<std::int64_t>(EventType::trading_halt)) {
    fail("event type is outside the documented range");
  }
  if (size < std::numeric_limits<std::int32_t>::min() ||
      size > std::numeric_limits<std::int32_t>::max()) {
    fail("size is outside int32 range");
  }
  if (price < std::numeric_limits<std::int32_t>::min() ||
      price > std::numeric_limits<std::int32_t>::max()) {
    fail("price is outside int32 range");
  }
  if (direction < std::numeric_limits<std::int8_t>::min() ||
      direction > std::numeric_limits<std::int8_t>::max()) {
    fail("direction is outside int8 range");
  }

  finish_message_row();
  message.event_type = static_cast<EventType>(event_type);
  message.size = static_cast<std::int32_t>(size);
  message.price = static_cast<std::int32_t>(price);
  message.direction = static_cast<std::int8_t>(direction);
  return message;
}

BookSnapshot CsvCursor::read_book() {
  if (empty()) {
    fail("unexpected end of file");
  }

  std::array<std::int64_t, 4 * kDepth> fields{};
  for (std::size_t index = 0; index < fields.size(); ++index) {
    fields[index] = read_integer();
    if (index + 1 < fields.size()) {
      consume_comma();
    }
  }

  for (const std::int64_t value : fields) {
    if (value < std::numeric_limits<std::int32_t>::min() ||
        value > std::numeric_limits<std::int32_t>::max()) {
      fail("book value is outside int32 range");
    }
  }

  finish_book_row();
  BookSnapshot snapshot;
  for (std::size_t level = 0; level < kDepth; ++level) {
    const std::size_t offset = 4 * level;
    snapshot.asks[level] = {
        static_cast<std::int32_t>(fields[offset]),
        static_cast<std::int32_t>(fields[offset + 1]),
    };
    snapshot.bids[level] = {
        static_cast<std::int32_t>(fields[offset + 2]),
        static_cast<std::int32_t>(fields[offset + 3]),
    };
  }
  return snapshot;
}

std::int64_t CsvCursor::read_integer() {
  if (cursor_ == end_) {
    fail("expected integer");
  }

  bool negative = false;
  if (*cursor_ == '-' || *cursor_ == '+') {
    negative = *cursor_ == '-';
    ++cursor_;
  }
  if (cursor_ == end_ || *cursor_ < '0' || *cursor_ > '9') {
    fail("expected integer digit");
  }

  std::uint64_t value = 0;
  while (cursor_ != end_ && *cursor_ >= '0' && *cursor_ <= '9') {
    const std::uint64_t digit = static_cast<std::uint64_t>(*cursor_ - '0');
    if (value > (std::numeric_limits<std::uint64_t>::max() - digit) / 10) {
      fail("integer overflow");
    }
    value = value * 10 + digit;
    ++cursor_;
  }

  constexpr std::uint64_t positive_limit =
      static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max());
  constexpr std::uint64_t negative_limit = positive_limit + 1;
  if ((!negative && value > positive_limit) ||
      (negative && value > negative_limit)) {
    fail("integer outside int64 range");
  }
  if (negative && value == negative_limit) {
    return std::numeric_limits<std::int64_t>::min();
  }

  const std::int64_t signed_value = static_cast<std::int64_t>(value);
  return negative ? -signed_value : signed_value;
}

std::uint64_t CsvCursor::read_timestamp_ns() {
  if (cursor_ == end_ || *cursor_ < '0' || *cursor_ > '9') {
    fail("expected timestamp");
  }

  std::uint64_t seconds = 0;
  while (cursor_ != end_ && *cursor_ >= '0' && *cursor_ <= '9') {
    const std::uint64_t digit = static_cast<std::uint64_t>(*cursor_ - '0');
    if (seconds > (std::numeric_limits<std::uint64_t>::max() - digit) / 10) {
      fail("timestamp overflow");
    }
    seconds = seconds * 10 + digit;
    ++cursor_;
  }

  std::uint64_t fractional = 0;
  std::uint32_t digits = 0;
  if (cursor_ != end_ && *cursor_ == '.') {
    ++cursor_;
    if (cursor_ == end_ || *cursor_ < '0' || *cursor_ > '9') {
      fail("expected timestamp fraction");
    }
    while (cursor_ != end_ && *cursor_ >= '0' && *cursor_ <= '9') {
      if (digits == 9) {
        fail("timestamp has more than nine fractional digits");
      }
      fractional =
          fractional * 10 + static_cast<std::uint64_t>(*cursor_ - '0');
      ++digits;
      ++cursor_;
    }
  }
  while (digits < 9) {
    fractional *= 10;
    ++digits;
  }

  constexpr std::uint64_t billion = 1'000'000'000;
  if (seconds >
      (std::numeric_limits<std::uint64_t>::max() - fractional) / billion) {
    fail("timestamp overflow");
  }
  return seconds * billion + fractional;
}

void CsvCursor::consume_comma() {
  if (cursor_ == end_ || *cursor_ != ',') {
    fail("expected comma");
  }
  ++cursor_;
}

void CsvCursor::finish_message_row() {
  if (cursor_ != end_ && *cursor_ == ',') {
    ++cursor_;
    while (cursor_ != end_ && *cursor_ != '\n' && *cursor_ != '\r') {
      if (*cursor_ == ',') {
        fail("message row has more than seven fields");
      }
      ++cursor_;
    }
  }
  consume_row_end();
}

void CsvCursor::finish_book_row() {
  consume_row_end();
}

void CsvCursor::consume_row_end() {
  if (cursor_ == end_) {
    return;
  }
  if (*cursor_ == '\r') {
    ++cursor_;
    if (cursor_ == end_ || *cursor_ != '\n') {
      fail("expected newline after carriage return");
    }
  }
  if (*cursor_ != '\n') {
    fail("unexpected data after final field");
  }
  ++cursor_;
  ++line_;
}

void CsvCursor::fail(const std::string& reason) const {
  std::ostringstream message;
  message << source_ << ':' << line_ << " (byte " << (cursor_ - begin_)
          << "): " << reason;
  throw ParseError(message.str());
}

std::vector<EventRecord> decode_paired_csv(
    std::string_view message_input,
    std::string_view book_input,
    std::string message_source,
    std::string book_source) {
  CsvCursor messages(message_input, message_source);
  CsvCursor books(book_input, book_source);
  std::vector<EventRecord> events;

  while (!messages.empty() && !books.empty()) {
    events.push_back({messages.read_message(), books.read_book()});
  }
  if (!messages.empty() || !books.empty()) {
    std::ostringstream error;
    error << "message/orderbook row-count mismatch near " << message_source
          << ':' << messages.line() << " and " << book_source << ':'
          << books.line();
    throw ParseError(error.str());
  }
  return events;
}

std::vector<EventRecord> decode_paired_files(
    const std::string& message_path,
    const std::string& book_path) {
  const MappedFile messages(message_path);
  const MappedFile books(book_path);
  return decode_paired_csv(
      messages.view(), books.view(), message_path, book_path);
}

}  // namespace tsla_lob
