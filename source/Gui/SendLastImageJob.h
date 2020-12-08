#pragma once

#include "Base/Job.h"

#include "Web/Definitions.h"

#include "Definitions.h"

class SendLastImageJob
    : public Job
{
    Q_OBJECT
public:
    SendLastImageJob(
        string const& currentSimulationId,
        string const& currentToken,
        string const& taskId,
        IntVector2D const& pos,
        IntVector2D const& size,
        SimulationAccess* simAccess,
        WebAccess* webAccess,
        QObject* parent);

    void process() override;
    bool isFinished() const override;
    bool isBlocking() const override;

private:
    void requestImage();
    void sendImageToServer();
    void finish();

    Q_SLOT void imageFromGpuReceived();
    Q_SLOT void serverReceivedImage();

    enum class State
    {
        Init,
        ImageFromGpuRequested,
        ImageToServeSent,
        Finished
    };

    bool _isReady = true;
    State _state = State::Init;

    IntVector2D _pos;
    IntVector2D _size;
    string _currentSimulationId;
    string _currentToken;

    QImagePtr _image;
    QBuffer* _buffer = nullptr;
    QByteArray _encodedImageData;

    std::mutex _mutex;

    SimulationAccess* _simAccess = nullptr;
    WebAccess* _webAccess = nullptr;
};
