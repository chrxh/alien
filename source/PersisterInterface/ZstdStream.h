#pragma once

#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <zstd.h>

// Lightweight, iostream-based wrapper around Zstandard: ostream/ofstream for
// compression, istream/ifstream for decompression. Lets call sites compress and
// decompress through ordinary std::ostream/std::istream interfaces.
namespace zstd
{
    inline constexpr int DefaultCompressionLevel = 13;

    // Worker count for multithreaded compression of large payloads. Capped because zstd
    // splits the input into a bounded number of jobs, so extra workers stay idle.
    inline int recommendedWorkerCount()
    {
        return static_cast<int>(std::min(std::thread::hardware_concurrency(), 8u));
    }

    class Exception : public std::ios_base::failure
    {
    public:
        explicit Exception(std::size_t code)
            : std::ios_base::failure(std::string("zstd: ") + ZSTD_getErrorName(code))
        {}
    };

    class ostreambuf : public std::streambuf
    {
    public:
        explicit ostreambuf(std::streambuf* sink, int level = DefaultCompressionLevel, int workers = 0)
            : _sink(sink)
            , _cctx(ZSTD_createCCtx())
            , _inBuffer(ZSTD_CStreamInSize())
            , _outBuffer(ZSTD_CStreamOutSize())
        {
            assert(_sink);
            if (!_cctx) {
                throw std::runtime_error("zstd: failed to create compression context");
            }
            if (ZSTD_isError(ZSTD_CCtx_setParameter(_cctx, ZSTD_c_compressionLevel, level))) {
                ZSTD_freeCCtx(_cctx);
                throw std::runtime_error("zstd: invalid compression level");
            }
            // Multithreaded compression is best-effort: if libzstd was built without thread
            // support the parameter is rejected and compression stays single-threaded.
            if (workers > 0) {
                ZSTD_CCtx_setParameter(_cctx, ZSTD_c_nbWorkers, workers);
            }
            setp(_inBuffer.data(), _inBuffer.data() + _inBuffer.size());
        }

        ostreambuf(ostreambuf const&) = delete;
        ostreambuf& operator=(ostreambuf const&) = delete;

        ~ostreambuf() override
        {
            if (!_failed) {
                try {
                    finish();
                } catch (...) {
                }
            }
            ZSTD_freeCCtx(_cctx);
        }

        int_type overflow(int_type c = traits_type::eof()) override
        {
            if (pptr() > pbase()) {
                if (compressPutArea(ZSTD_e_continue) != 0) {
                    _failed = true;
                    setp(nullptr, nullptr);
                    return traits_type::eof();
                }
                _frameActive = true;
            }
            setp(_inBuffer.data(), _inBuffer.data() + _inBuffer.size());
            if (!traits_type::eq_int_type(c, traits_type::eof())) {
                return sputc(traits_type::to_char_type(c));
            }
            return traits_type::not_eof(c);
        }

        int sync() override { return finish(); }

    private:
        int compressPutArea(ZSTD_EndDirective mode)
        {
            ZSTD_inBuffer input{pbase(), static_cast<std::size_t>(pptr() - pbase()), 0};
            for (;;) {
                ZSTD_outBuffer output{_outBuffer.data(), _outBuffer.size(), 0};
                std::size_t const remaining = ZSTD_compressStream2(_cctx, &output, &input, mode);
                if (ZSTD_isError(remaining)) {
                    return -1;
                }
                if (output.pos > 0) {
                    auto const count = static_cast<std::streamsize>(output.pos);
                    if (_sink->sputn(_outBuffer.data(), count) != count) {
                        return -1;
                    }
                }
                bool const done = (mode == ZSTD_e_end) ? (remaining == 0) : (input.pos == input.size);
                if (done) {
                    break;
                }
            }
            return 0;
        }

        // Finalizes the current frame (ZSTD_e_end) and flushes the sink. Guards against
        // repeated calls (explicit flush plus destructor) producing empty trailing frames.
        int finish()
        {
            if (pptr() > pbase() || _frameActive) {
                if (compressPutArea(ZSTD_e_end) != 0) {
                    _failed = true;
                    return -1;
                }
                _frameActive = false;
                setp(_inBuffer.data(), _inBuffer.data() + _inBuffer.size());
            }
            return _sink->pubsync();
        }

        std::streambuf* _sink;
        ZSTD_CCtx* _cctx;
        std::vector<char> _inBuffer;
        std::vector<char> _outBuffer;
        bool _frameActive = false;
        bool _failed = false;
    };

    class istreambuf : public std::streambuf
    {
    public:
        explicit istreambuf(std::streambuf* source)
            : _source(source)
            , _dctx(ZSTD_createDCtx())
            , _inBuffer(ZSTD_DStreamInSize())
            , _outBuffer(ZSTD_DStreamOutSize())
            , _input{_inBuffer.data(), 0, 0}
        {
            assert(_source);
            if (!_dctx) {
                throw std::runtime_error("zstd: failed to create decompression context");
            }
            setg(_outBuffer.data(), _outBuffer.data(), _outBuffer.data());
        }

        istreambuf(istreambuf const&) = delete;
        istreambuf& operator=(istreambuf const&) = delete;

        ~istreambuf() override { ZSTD_freeDCtx(_dctx); }

        int_type underflow() override
        {
            if (gptr() < egptr()) {
                return traits_type::to_int_type(*gptr());
            }
            ZSTD_outBuffer output{_outBuffer.data(), _outBuffer.size(), 0};
            while (output.pos == 0) {
                if (_input.pos == _input.size) {
                    std::streamsize const count = _source->sgetn(_inBuffer.data(), static_cast<std::streamsize>(_inBuffer.size()));
                    if (count <= 0) {
                        break;
                    }
                    _input.src = _inBuffer.data();
                    _input.size = static_cast<std::size_t>(count);
                    _input.pos = 0;
                }
                std::size_t const ret = ZSTD_decompressStream(_dctx, &output, &_input);
                if (ZSTD_isError(ret)) {
                    throw Exception(ret);
                }
            }
            setg(_outBuffer.data(), _outBuffer.data(), _outBuffer.data() + output.pos);
            return output.pos == 0 ? traits_type::eof() : traits_type::to_int_type(*gptr());
        }

    private:
        std::streambuf* _source;
        ZSTD_DCtx* _dctx;
        std::vector<char> _inBuffer;
        std::vector<char> _outBuffer;
        ZSTD_inBuffer _input;
    };

    class ostream : public std::ostream
    {
    public:
        explicit ostream(std::ostream& os, int level = DefaultCompressionLevel, int workers = 0)
            : std::ostream(new ostreambuf(os.rdbuf(), level, workers))
        {
            exceptions(std::ios_base::badbit);
        }
        explicit ostream(std::streambuf* sink, int level = DefaultCompressionLevel, int workers = 0)
            : std::ostream(new ostreambuf(sink, level, workers))
        {
            exceptions(std::ios_base::badbit);
        }
        ~ostream() override { delete rdbuf(); }
    };

    class istream : public std::istream
    {
    public:
        explicit istream(std::istream& is)
            : std::istream(new istreambuf(is.rdbuf()))
        {
            exceptions(std::ios_base::badbit);
        }
        explicit istream(std::streambuf* source)
            : std::istream(new istreambuf(source))
        {
            exceptions(std::ios_base::badbit);
        }
        ~istream() override { delete rdbuf(); }
    };

    namespace detail
    {
        template <typename FStream>
        struct FStreamHolder
        {
            FStreamHolder(std::string const& filename, std::ios_base::openmode mode)
                : _fs(filename, mode)
            {}
            FStream _fs;
        };
    }

    class ofstream
        : private detail::FStreamHolder<std::ofstream>
        , public std::ostream
    {
    public:
        explicit ofstream(std::string const& filename, std::ios_base::openmode mode = std::ios_base::out, int level = DefaultCompressionLevel, int workers = 0)
            : detail::FStreamHolder<std::ofstream>(filename, mode | std::ios_base::binary)
            , std::ostream(new ostreambuf(_fs.rdbuf(), level, workers))
        {
            exceptions(std::ios_base::badbit);
            if (!_fs.is_open()) {
                setstate(std::ios_base::failbit);
            }
        }
        void close()
        {
            std::ostream::flush();
            _fs.close();
        }
        bool is_open() const { return _fs.is_open(); }
        ~ofstream() override
        {
            std::ostream::flush();
            delete rdbuf();
        }
    };

    class ifstream
        : private detail::FStreamHolder<std::ifstream>
        , public std::istream
    {
    public:
        explicit ifstream(std::string const& filename, std::ios_base::openmode mode = std::ios_base::in)
            : detail::FStreamHolder<std::ifstream>(filename, mode | std::ios_base::binary)
            , std::istream(new istreambuf(_fs.rdbuf()))
        {
            exceptions(std::ios_base::badbit);
            if (!_fs.is_open()) {
                setstate(std::ios_base::failbit);
            }
        }
        void close() { _fs.close(); }
        bool is_open() const { return _fs.is_open(); }
        ~ifstream() override { delete rdbuf(); }
    };
}
