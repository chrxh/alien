// Copyright (c) 2017-2025, University of Cincinnati, developed by Henry Schreiner
// under NSF AWARD 1414736 and by the respective contributors.
// All rights reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

// IWYU pragma: private, include "CLI/CLI.hpp"

#include "Error.hpp"
#include "Macros.hpp"
#include "StringTools.hpp"
#include "TypeTools.hpp"

// [CLI11:public_includes:set]
#include <cmath>
#include <cstdint>
#include <functional>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>
// [CLI11:public_includes:end]

// [CLI11:validators_hpp_filesystem:verbatim]

#if defined CLI11_HAS_FILESYSTEM && CLI11_HAS_FILESYSTEM > 0
#include <filesystem>  // NOLINT(build/include)
#else
#include <sys/stat.h>
#include <sys/types.h>
#endif

// [CLI11:validators_hpp_filesystem:end]

namespace CLI {
// [CLI11:validators_hpp:verbatim]

class Option;

/// @defgroup validator_group Validators

/// @brief Some validators that are provided
///
/// These are simple `std::string(const std::string&)` validators that are useful. They return
/// a string if the validation fails. A custom struct is provided, as well, with the same user
/// semantics, but with the ability to provide a new type name.
/// @{

///
class Validator {
  protected:
    /// This is the description function, if empty the description_ will be used
    std::function<std::string()> desc_function_{[]() { return std::string{}; }};

    /// This is the base function that is to be called.
    /// Returns a string error message if validation fails.
    std::function<std::string(std::string &)> func_{[](std::string &) { return std::string{}; }};
    /// The name for search purposes of the Validator
    std::string name_{};
    /// A Validator will only apply to an indexed value (-1 is all elements)
    int application_index_ = -1;
    /// Enable for Validator to allow it to be disabled if need be
    bool active_{true};
    /// specify that a validator should not modify the input
    bool non_modifying_{false};

    Validator(std::string validator_desc, std::function<std::string(std::string &)> func)
        : desc_function_([validator_desc]() { return validator_desc; }), func_(std::move(func)) {}

  public:
    Validator() = default;
    /// Construct a Validator with just the description string
    explicit Validator(std::string validator_desc) : desc_function_([validator_desc]() { return validator_desc; }) {}
    /// Construct Validator from basic information
    Validator(std::function<std::string(std::string &)> op, std::string validator_desc, std::string validator_name = "")
        : desc_function_([validator_desc]() { return validator_desc; }), func_(std::move(op)),
          name_(std::move(validator_name)) {}
    /// Set the Validator operation function
    Validator &operation(std::function<std::string(std::string &)> op) {
        func_ = std::move(op);
        return *this;
    }
    /// This is the required operator for a Validator - provided to help
    /// users (CLI11 uses the member `func` directly)
    std::string operator()(std::string &str) const;

    /// This is the required operator for a Validator - provided to help
    /// users (CLI11 uses the member `func` directly)
    std::string operator()(const std::string &str) const {
        std::string value = str;
        return (active_) ? func_(value) : std::string{};
    }

    /// Specify the type string
    Validator &description(std::string validator_desc) {
        desc_function_ = [validator_desc]() { return validator_desc; };
        return *this;
    }
    /// Specify the type string
    CLI11_NODISCARD Validator description(std::string validator_desc) const;

    /// Generate type description information for the Validator
    CLI11_NODISCARD std::string get_description() const {
        if(active_) {
            return desc_function_();
        }
        return std::string{};
    }
    /// Specify the type string
    Validator &name(std::string validator_name) {
        name_ = std::move(validator_name);
        return *this;
    }
    /// Specify the type string
    CLI11_NODISCARD Validator name(std::string validator_name) const {
        Validator newval(*this);
        newval.name_ = std::move(validator_name);
        return newval;
    }
    /// Get the name of the Validator
    CLI11_NODISCARD const std::string &get_name() const { return name_; }
    /// Specify whether the Validator is active or not
    Validator &active(bool active_val = true) {
        active_ = active_val;
        return *this;
    }
    /// Specify whether the Validator is active or not
    CLI11_NODISCARD Validator active(bool active_val = true) const {
        Validator newval(*this);
        newval.active_ = active_val;
        return newval;
    }

    /// Specify whether the Validator can be modifying or not
    Validator &non_modifying(bool no_modify = true) {
        non_modifying_ = no_modify;
        return *this;
    }
    /// Specify the application index of a validator
    Validator &application_index(int app_index) {
        application_index_ = app_index;
        return *this;
    }
    /// Specify the application index of a validator
    CLI11_NODISCARD Validator application_index(int app_index) const {
        Validator newval(*this);
        newval.application_index_ = app_index;
        return newval;
    }
    /// Get the current value of the application index
    CLI11_NODISCARD int get_application_index() const { return application_index_; }
    /// Get a boolean if the validator is active
    CLI11_NODISCARD bool get_active() const { return active_; }

    /// Get a boolean if the validator is allowed to modify the input returns true if it can modify the input
    CLI11_NODISCARD bool get_modifying() const { return !non_modifying_; }

    /// Combining validators is a new validator. Type comes from left validator if function, otherwise only set if the
    /// same.
    Validator operator&(const Validator &other) const;

    /// Combining validators is a new validator. Type comes from left validator if function, otherwise only set if the
    /// same.
    Validator operator|(const Validator &other) const;

    /// Create a validator that fails when a given validator succeeds
    Validator operator!() const;

  private:
    void _merge_description(const Validator &val1, const Validator &val2, const std::string &merger);
};

/// Alias for Validator for custom Validator for clarity
using CustomValidator = Validator;

// The implementation of the built in validators is using the Validator class;
// the user is only expected to use the const (static) versions (since there's no setup).
// Therefore, this is in detail.
namespace detail {

/// CLI enumeration of different file types
enum class path_type : std::uint8_t { nonexistent, file, directory };

/// get the type of the path from a file name
CLI11_INLINE path_type check_path(const char *file) noexcept;

// Static is not needed here, because global const implies static.

/// Check for an existing file (returns error message if check fails)
class ExistingFileValidator : public Validator {
  public:
    ExistingFileValidator();
};

/// Check for an existing directory (returns error message if check fails)
class ExistingDirectoryValidator : public Validator {
  public:
    ExistingDirectoryValidator();
};

/// Check for an existing path
class ExistingPathValidator : public Validator {
  public:
    ExistingPathValidator();
};

/// Check for an non-existing path
class NonexistentPathValidator : public Validator {
  public:
    NonexistentPathValidator();
};

class EscapedStringTransformer : public Validator {
  public:
    EscapedStringTransformer();
};

}  // namespace detail

/// Check for existing file (returns error message if check fails)
const detail::ExistingFileValidator ExistingFile;

/// Check for an existing directory (returns error message if check fails)
const detail::ExistingDirectoryValidator ExistingDirectory;

/// Check for an existing path
const detail::ExistingPathValidator ExistingPath;

/// Check for an non-existing path
const detail::NonexistentPathValidator NonexistentPath;

/// convert escaped characters into their associated values
const detail::EscapedStringTransformer EscapedString;

/// Modify a path if the file is a particular default location, can be used as Check or transform
/// with the error return optionally disabled
class FileOnDefaultPath : public Validator {
  public:
    explicit FileOnDefaultPath(std::string default_path, bool enableErrorReturn = true);
};

/// Produce a range (factory). Min and max are inclusive.
class Range : public Validator {
  public:
    /// This produces a range with min and max inclusive.
    ///
    /// Note that the constructor is templated, but the struct is not, so C++17 is not
    /// needed to provide nice syntax for Range(a,b).
    template <typename T>
    Range(T min_val, T max_val, const std::string &validator_name = std::string{}) : Validator(validator_name) {
        if(validator_name.empty()) {
            std::stringstream out;
            out << detail::type_name<T>() << " in [" << min_val << " - " << max_val << "]";
            description(out.str());
        }

        func_ = [min_val, max_val](std::string &input) {
            using CLI::detail::lexical_cast;
            T val;
            bool converted = lexical_cast(input, val);
            if((!converted) || (val < min_val || val > max_val)) {
                std::stringstream out;
                out << "Value " << input << " not in range [";
                out << min_val << " - " << max_val << "]";
                return out.str();
            }
            return std::string{};
        };
    }

    /// Range of one value is 0 to value
    template <typename T>
    explicit Range(T max_val, const std::string &validator_name = std::string{})
        : Range(static_cast<T>(0), max_val, validator_name) {}
};

/// Check for a non negative number
const Range NonNegativeNumber((std::numeric_limits<double>::max)(), "NONNEGATIVE");

/// Check for a positive valued number (val>0.0), <double>::min  here is the smallest positive number
const Range PositiveNumber((std::numeric_limits<double>::min)(), (std::numeric_limits<double>::max)(), "POSITIVE");

namespace detail {
// the following suggestion was made by Nikita Ofitserov(@himikof)
// done in templates to prevent compiler warnings on negation of unsigned numbers

/// Do a check for overflow on signed numbers
template <typename T>
inline typename std::enable_if<std::is_signed<T>::value, T>::type overflowCheck(const T &a, const T &b) {
    if((a > 0) == (b > 0)) {
        return ((std::numeric_limits<T>::max)() / (std::abs)(a) < (std::abs)(b));
    }
    return ((std::numeric_limits<T>::min)() / (std::abs)(a) > -(std::abs)(b));
}
/// Do a check for overflow on unsigned numbers
template <typename T>
inline typename std::enable_if<!std::is_signed<T>::value, T>::type overflowCheck(const T &a, const T &b) {
    return ((std::numeric_limits<T>::max)() / a < b);
}

/// Performs a *= b; if it doesn't cause integer overflow. Returns false otherwise.
template <typename T> typename std::enable_if<std::is_integral<T>::value, bool>::type checked_multiply(T &a, T b) {
    if(a == 0 || b == 0 || a == 1 || b == 1) {
        a *= b;
        return true;
    }
    if(a == (std::numeric_limits<T>::min)() || b == (std::numeric_limits<T>::min)()) {
        return false;
    }
    if(overflowCheck(a, b)) {
        return false;
    }
    a *= b;
    return true;
}

/// Performs a *= b; if it doesn't equal infinity. Returns false otherwise.
template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, bool>::type checked_multiply(T &a, T b) {
    T c = a * b;
    if(std::isinf(c) && !std::isinf(a) && !std::isinf(b)) {
        return false;
    }
    a = c;
    return true;
}
/// Split a string into a program name and command line arguments
/// the string is assumed to contain a file name followed by other arguments
/// the return value contains is a pair with the first argument containing the program name and the second
/// everything else.
CLI11_INLINE std::pair<std::string, std::string> split_program_name(std::string commandline);

}  // namespace detail
/// @}

// [CLI11:validators_hpp:end]
}  // namespace CLI

#ifndef CLI11_COMPILE
#include "impl/Validators_inl.hpp"  // IWYU pragma: export
#endif
