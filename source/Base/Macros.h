#pragma once

#include <stdexcept>

#define THROW_NOT_IMPLEMENTED() throw std::runtime_error("not implemented")

#define CHECK(expression) \
    if (!(expression)) { \
        throw std::runtime_error("check failed"); \
    }

#define MEMBER(className, type, name, initialValue) \
    type _##name = initialValue; \
    className& name(type const& name) \
    { \
        _##name = name; \
        return *this; \
    } \
    className& name(type&& name) \
    { \
        _##name = std::move(name); \
        return *this; \
    }

#define SETTER_SHARED_PTR(className, type, name) \
    className& name(_##type const& name) \
    { \
        _##name = std::make_shared<_##type>(name); \
        return *this; \
    }

#define SETTER(className, type, name) \
    className& name(type const& name) \
    { \
        _##name = name; \
        return *this; \
    }
