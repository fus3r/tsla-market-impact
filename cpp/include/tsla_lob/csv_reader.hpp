#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "tsla_lob/lobster.hpp"

namespace tsla_lob {

class ParseError : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

class MappedFile {
 public:
  explicit MappedFile(const std::string& path);
  ~MappedFile();

  MappedFile(const MappedFile&) = delete;
  MappedFile& operator=(const MappedFile&) = delete;
  MappedFile(MappedFile&& other) noexcept;
  MappedFile& operator=(MappedFile&& other) noexcept;

  [[nodiscard]] std::string_view view() const noexcept;
  [[nodiscard]] std::uint64_t size() const noexcept;

 private:
  void close() noexcept;

  int descriptor_{-1};
  const char* data_{nullptr};
  std::size_t size_{0};
};

class CsvCursor {
 public:
  explicit CsvCursor(std::string_view input, std::string source = "<memory>");

  [[nodiscard]] bool empty() const noexcept;
  [[nodiscard]] std::uint64_t line() const noexcept;
  [[nodiscard]] Message read_message();
  [[nodiscard]] BookSnapshot read_book();

 private:
  [[nodiscard]] std::int64_t read_integer();
  [[nodiscard]] std::uint64_t read_timestamp_ns();
  void consume_comma();
  void finish_message_row();
  void finish_book_row();
  void consume_row_end();
  [[noreturn]] void fail(const std::string& reason) const;

  const char* begin_;
  const char* cursor_;
  const char* end_;
  std::string source_;
  std::uint64_t line_{1};
};

[[nodiscard]] std::vector<EventRecord> decode_paired_csv(
    std::string_view message_input,
    std::string_view book_input,
    std::string message_source = "<messages>",
    std::string book_source = "<orderbook>");

[[nodiscard]] std::vector<EventRecord> decode_paired_files(
    const std::string& message_path,
    const std::string& book_path);

}  // namespace tsla_lob
