#include "VideoWriter.h"

#include <format>
#include <future>
#include <memory>
#include <span>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>

#include <objbase.h>

#include <codecapi.h>
#include <mfapi.h>
#include <mfidl.h>
#include <mfreadwrite.h>
#include <wrl/client.h>
#else
#include <csignal>
#include <cstdio>
#include <cstdlib>
#endif

namespace
{
    // Frames that wait for the encoding take a lot of memory
    auto constexpr MaxPendingFrames = size_t{4};

    class Encoder
    {
    public:
        virtual ~Encoder() = default;

        virtual void addFrame(RgbImage const& image) = 0;
        virtual void finish() = 0;
    };

#ifdef _WIN32
    void checkResult(HRESULT result, std::string const& operation)
    {
        if (FAILED(result)) {
            throw AlienException(std::format("{} failed with error {:#010x}.", operation, static_cast<uint32_t>(result)));
        }
    }

    // Windows N editions lack Media Foundation unless the Media Feature Pack is installed. Its functions are therefore looked up at run time
    // instead of being linked.
    struct MediaFoundationApi
    {
        decltype(&MFStartup) startup = nullptr;
        decltype(&MFShutdown) shutdown = nullptr;
        decltype(&MFCreateAttributes) createAttributes = nullptr;
        decltype(&MFCreateMediaType) createMediaType = nullptr;
        decltype(&MFCreateMemoryBuffer) createMemoryBuffer = nullptr;
        decltype(&MFCreateSample) createSample = nullptr;
        decltype(&MFCreateSinkWriterFromURL) createSinkWriterFromUrl = nullptr;
    };

    template <typename Function>
    void loadFunction(HMODULE module, char const* name, Function& function)
    {
        function = reinterpret_cast<Function>(GetProcAddress(module, name));
        if (function == nullptr) {
            throw AlienException(std::format("Media Foundation does not provide {}.", name));
        }
    }

    MediaFoundationApi loadMediaFoundationApi()
    {
        auto platformModule = LoadLibraryExW(L"mfplat.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        auto readWriteModule = LoadLibraryExW(L"mfreadwrite.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (platformModule == nullptr || readWriteModule == nullptr) {
            throw AlienException("Media Foundation is not available. Windows N editions need the Media Feature Pack for recording videos.");
        }
        MediaFoundationApi result;
        loadFunction(platformModule, "MFStartup", result.startup);
        loadFunction(platformModule, "MFShutdown", result.shutdown);
        loadFunction(platformModule, "MFCreateAttributes", result.createAttributes);
        loadFunction(platformModule, "MFCreateMediaType", result.createMediaType);
        loadFunction(platformModule, "MFCreateMemoryBuffer", result.createMemoryBuffer);
        loadFunction(platformModule, "MFCreateSample", result.createSample);
        loadFunction(readWriteModule, "MFCreateSinkWriterFromURL", result.createSinkWriterFromUrl);
        return result;
    }

    // Media Foundation objects are created and released by the thread that holds the session
    class MediaFoundationSession
    {
    public:
        MediaFoundationSession()
            : _api(loadMediaFoundationApi())
        {
            checkResult(CoInitializeEx(nullptr, COINIT_MULTITHREADED), "Initializing COM");
            auto result = _api.startup(MF_VERSION, MFSTARTUP_FULL);
            if (FAILED(result)) {
                CoUninitialize();
                checkResult(result, "Starting Media Foundation");
            }
        }

        ~MediaFoundationSession()
        {
            _api.shutdown();
            CoUninitialize();
        }

        MediaFoundationApi const& getApi() const { return _api; }

    private:
        MediaFoundationApi _api;
    };

    class MediaFoundationEncoder : public Encoder
    {
    public:
        MediaFoundationEncoder(std::filesystem::path const& filename, IntVector2D const& resolution, int fps, int bitRate)
            : _resolution(resolution)
            , _fps(fps)
        {
            try {
                createSinkWriter(filename, bitRate, true);
            } catch (AlienException const&) {
                // Hardware encoders do not support every configuration
                _sinkWriter.Reset();
                createSinkWriter(filename, bitRate, false);
            }
        }

        void addFrame(RgbImage const& image) override
        {
            auto const& api = _session.getApi();
            auto numPixels = static_cast<size_t>(_resolution.x) * _resolution.y;
            auto size = toUInt32(numPixels * 4);

            Microsoft::WRL::ComPtr<IMFMediaBuffer> buffer;
            checkResult(api.createMemoryBuffer(size, &buffer), "Creating a video frame");
            BYTE* data = nullptr;
            checkResult(buffer->Lock(&data, nullptr, nullptr), "Accessing a video frame");

            // RGB32 orders the colors as blue, green and red, followed by an unused byte
            auto source = std::span(image.pixels);
            auto destination = std::span(data, size);
            for (size_t index = 0; index < numPixels; ++index) {
                destination[index * 4] = source[index * 3 + 2];
                destination[index * 4 + 1] = source[index * 3 + 1];
                destination[index * 4 + 2] = source[index * 3];
                destination[index * 4 + 3] = 255;
            }
            checkResult(buffer->Unlock(), "Accessing a video frame");
            checkResult(buffer->SetCurrentLength(size), "Filling a video frame");

            Microsoft::WRL::ComPtr<IMFSample> sample;
            checkResult(api.createSample(&sample), "Creating a video sample");
            checkResult(sample->AddBuffer(buffer.Get()), "Filling a video sample");
            auto startTime = calcFrameTime(_numFrames);
            checkResult(sample->SetSampleTime(startTime), "Timing a video sample");
            checkResult(sample->SetSampleDuration(calcFrameTime(_numFrames + 1) - startTime), "Timing a video sample");
            checkResult(_sinkWriter->WriteSample(_streamIndex, sample.Get()), "Encoding a video frame");
            ++_numFrames;
        }

        void finish() override { checkResult(_sinkWriter->Finalize(), "Finishing the video"); }

    private:
        void createSinkWriter(std::filesystem::path const& filename, int bitRate, bool hardwareEncoding)
        {
            auto const& api = _session.getApi();

            Microsoft::WRL::ComPtr<IMFAttributes> attributes;
            checkResult(api.createAttributes(&attributes, 1), "Configuring the video file");
            checkResult(attributes->SetUINT32(MF_READWRITE_ENABLE_HARDWARE_TRANSFORMS, hardwareEncoding ? TRUE : FALSE), "Configuring the video file");
            checkResult(api.createSinkWriterFromUrl(filename.wstring().c_str(), nullptr, attributes.Get(), &_sinkWriter), "Creating the video file");

            Microsoft::WRL::ComPtr<IMFMediaType> outputType;
            checkResult(api.createMediaType(&outputType), "Configuring the H.264 encoding");
            checkResult(outputType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video), "Configuring the H.264 encoding");
            checkResult(outputType->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_H264), "Configuring the H.264 encoding");
            checkResult(outputType->SetUINT32(MF_MT_AVG_BITRATE, toUInt32(bitRate)), "Configuring the H.264 encoding");
            checkResult(outputType->SetUINT32(MF_MT_MPEG2_PROFILE, eAVEncH264VProfile_High), "Configuring the H.264 encoding");
            setVideoAttributes(outputType.Get());
            checkResult(_sinkWriter->AddStream(outputType.Get(), &_streamIndex), "Configuring the H.264 encoding");

            // A positive stride orders the rows from top to bottom
            Microsoft::WRL::ComPtr<IMFMediaType> inputType;
            checkResult(api.createMediaType(&inputType), "Configuring the video input");
            checkResult(inputType->SetGUID(MF_MT_MAJOR_TYPE, MFMediaType_Video), "Configuring the video input");
            checkResult(inputType->SetGUID(MF_MT_SUBTYPE, MFVideoFormat_RGB32), "Configuring the video input");
            checkResult(inputType->SetUINT32(MF_MT_DEFAULT_STRIDE, toUInt32(_resolution.x * 4)), "Configuring the video input");
            setVideoAttributes(inputType.Get());
            checkResult(_sinkWriter->SetInputMediaType(_streamIndex, inputType.Get(), nullptr), "Configuring the video input");

            checkResult(_sinkWriter->BeginWriting(), "Starting the video encoding");
        }

        void setVideoAttributes(IMFMediaType* mediaType) const
        {
            checkResult(mediaType->SetUINT32(MF_MT_INTERLACE_MODE, MFVideoInterlace_Progressive), "Configuring the video");
            checkResult(MFSetAttributeSize(mediaType, MF_MT_FRAME_SIZE, toUInt32(_resolution.x), toUInt32(_resolution.y)), "Configuring the video");
            checkResult(MFSetAttributeRatio(mediaType, MF_MT_FRAME_RATE, toUInt32(_fps), 1), "Configuring the video");
            checkResult(MFSetAttributeRatio(mediaType, MF_MT_PIXEL_ASPECT_RATIO, 1, 1), "Configuring the video");
        }

        // In units of 100 nanoseconds
        LONGLONG calcFrameTime(int64_t frame) const { return frame * 10000000 / _fps; }

        MediaFoundationSession _session;
        Microsoft::WRL::ComPtr<IMFSinkWriter> _sinkWriter;
        DWORD _streamIndex = 0;
        IntVector2D _resolution;
        int _fps = 0;
        int64_t _numFrames = 0;
    };
#else
    std::string quoteForShell(std::string const& text)
    {
        std::string result = "'";
        for (auto character : text) {
            if (character == '\'') {
                result += "'\\''";
            } else {
                result += character;
            }
        }
        return result + "'";
    }

    class FfmpegEncoder : public Encoder
    {
    public:
        FfmpegEncoder(std::filesystem::path const& filename, IntVector2D const& resolution, int fps, int bitRate)
        {
            // A terminated ffmpeg should lead to a write error instead of ending the program
            std::signal(SIGPIPE, SIG_IGN);

            if (std::system("ffmpeg -version > /dev/null 2>&1") != 0) {
                throw AlienException("Recording videos requires ffmpeg, which could not be found.");
            }
            auto command = std::format(
                "ffmpeg -loglevel error -y -f rawvideo -pix_fmt rgb24 -video_size {}x{} -framerate {} -i - -pix_fmt yuv420p -b:v {} {}",
                resolution.x,
                resolution.y,
                fps,
                bitRate,
                quoteForShell(filename.string()));
            _pipe = popen(command.c_str(), "w");
            if (_pipe == nullptr) {
                throw AlienException("ffmpeg could not be started.");
            }
        }

        ~FfmpegEncoder() override
        {
            if (_pipe != nullptr) {
                pclose(_pipe);
            }
        }

        void addFrame(RgbImage const& image) override
        {
            if (std::fwrite(image.pixels.data(), 1, image.pixels.size(), _pipe) != image.pixels.size()) {
                throw AlienException("ffmpeg did not accept the video frames.");
            }
        }

        void finish() override
        {
            auto status = pclose(_pipe);
            _pipe = nullptr;
            if (status != 0) {
                throw AlienException("ffmpeg could not write the video.");
            }
        }

    private:
        FILE* _pipe = nullptr;
    };
#endif

    std::unique_ptr<Encoder> createEncoder(std::filesystem::path const& filename, IntVector2D const& resolution, int fps, int bitRate)
    {
        if (resolution.x <= 0 || resolution.y <= 0 || resolution.x % 2 != 0 || resolution.y % 2 != 0) {
            throw AlienException("The width and height of a video must be positive and even.");
        }
        if (fps <= 0 || bitRate <= 0) {
            throw AlienException("The frame rate and the bit rate of a video must be positive.");
        }
#ifdef _WIN32
        return std::make_unique<MediaFoundationEncoder>(filename, resolution, fps, bitRate);
#else
        return std::make_unique<FfmpegEncoder>(filename, resolution, fps, bitRate);
#endif
    }
}

_VideoWriter::_VideoWriter(std::filesystem::path const& filename, IntVector2D const& resolution, int fps, int bitRate)
{
    std::promise<void> encoderCreated;
    auto encoderCreation = encoderCreated.get_future();
    _thread = std::thread([this, filename, resolution, fps, bitRate, encoderCreated = std::move(encoderCreated)]() mutable {
        std::unique_ptr<Encoder> encoder;
        try {
            encoder = createEncoder(filename, resolution, fps, bitRate);
            encoderCreated.set_value();
        } catch (...) {
            encoderCreated.set_exception(std::current_exception());
            return;
        }

        // After an error, the remaining frames are discarded
        while (auto frame = waitForFrame()) {
            if (!getError()) {
                try {
                    encoder->addFrame(*frame);
                } catch (std::exception const& exception) {
                    setError(exception.what());
                }
            }
        }
        if (!getError()) {
            try {
                encoder->finish();
            } catch (std::exception const& exception) {
                setError(exception.what());
            }
        }
    });

    try {
        encoderCreation.get();
    } catch (...) {
        _thread.join();
        throw;
    }
}

_VideoWriter::~_VideoWriter()
{
    stopEncoding();
}

void _VideoWriter::addFrame(RgbImage&& image)
{
    {
        std::unique_lock lock(_mutex);
        _condition.wait(lock, [this] { return _frames.size() < MaxPendingFrames || _error.has_value(); });
        if (_error) {
            return;
        }
        _frames.emplace_back(std::move(image));
    }
    _condition.notify_all();
}

void _VideoWriter::finish()
{
    stopEncoding();
    if (auto error = getError()) {
        throw AlienException(*error);
    }
}

std::optional<std::string> _VideoWriter::getError() const
{
    std::lock_guard lock(_mutex);
    return _error;
}

std::optional<RgbImage> _VideoWriter::waitForFrame()
{
    std::optional<RgbImage> result;
    {
        std::unique_lock lock(_mutex);
        _condition.wait(lock, [this] { return !_frames.empty() || _finishing; });
        if (_frames.empty()) {
            return std::nullopt;
        }
        result = std::move(_frames.front());
        _frames.pop_front();
    }
    _condition.notify_all();
    return result;
}

void _VideoWriter::setError(std::string const& error)
{
    {
        std::lock_guard lock(_mutex);
        if (!_error) {
            _error = error;
        }
    }
    _condition.notify_all();
}

void _VideoWriter::stopEncoding()
{
    if (!_thread.joinable()) {
        return;
    }
    {
        std::lock_guard lock(_mutex);
        _finishing = true;
    }
    _condition.notify_all();
    _thread.join();
}
