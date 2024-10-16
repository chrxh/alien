#pragma once

#define MAKE_SINGLETON(ClassName) \
public: \
    static ClassName& get() \
    { \
        static ClassName instance; \
        return instance; \
    } \
\
private: \
    ClassName() = default; \
    ~ClassName() = default; \
    ClassName(ClassName const&) = delete; \
    ClassName& operator=(ClassName const&) = delete; \
    ClassName(ClassName&&) = delete; \
    ClassName& operator=(ClassName&&) = delete

#define MAKE_SINGLETON_NO_DEFAULT_CONSTRUCTOR(ClassName) \
public: \
    static ClassName& get() \
    { \
        static ClassName instance; \
        return instance; \
    } \
\
private: \
    ~ClassName() = default; \
    ClassName(ClassName const&) = delete; \
    ClassName& operator=(ClassName const&) = delete; \
    ClassName(ClassName&&) = delete; \
    ClassName& operator=(ClassName&&) = delete
