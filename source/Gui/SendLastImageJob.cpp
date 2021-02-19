#include "SendLastImageJob.h"

#include <iostream>
#include <sstream>
#include <QBuffer>
#include <QImage>

#include "Base/ServiceLocator.h"
#include "Base/LoggingService.h"

#include "EngineInterface/SimulationAccess.h"

#include "Web/WebAccess.h"

SendLastImageJob::SendLastImageJob(
    string const& currentSimulationId,
    string const& currentToken,
    IntVector2D const& pos,
    IntVector2D const& size,
    SimulationAccess* simAccess,
    WebAccess* webAccess,
    QObject* parent)
    : Job("LastImageJob", parent)
    , _currentSimulationId(currentSimulationId)
    , _currentToken(currentToken)
    , _pos(pos)
    , _size(size)
    , _simAccess(simAccess)
    , _webAccess(webAccess)
{
    connect(_simAccess, &SimulationAccess::imageReady, this, &SendLastImageJob::imageFromGpuReceived);
    connect(_webAccess, &WebAccess::sendLastImageReceived, this, &SendLastImageJob::serverReceivedImage);
}

void SendLastImageJob::process()
{
    if (!_isReady) {
        return;
    }

    switch (_state)
    {
    case State::Init:
        requestImage();
        break;
    case State::ImageFromGpuRequested:
        sendImageToServer();
        break;
    case State::ImageToServerSent:
        finish();
        break;
    default:
        break;
    }
}

bool SendLastImageJob::isFinished() const
{
    return State::Finished == _state;
}

bool SendLastImageJob::isBlocking() const
{
    return true;
}

void SendLastImageJob::requestImage()
{
    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();

    std::stringstream stream;
    stream << "Web: get last image with size " << _size.x << " x " << _size.y;
    loggingService->logMessage(Priority::Important, stream.str());

    _image = boost::make_shared<QImage>(_size.x, _size.y, QImage::Format_RGB32);
    auto const rect = IntRect{ _pos, IntVector2D{ _pos.x + _size.x, _pos.y + _size.y } };
    _simAccess->requirePixelImage(rect, _image, _mutex);

    _state = State::ImageFromGpuRequested;
    _isReady = false;
}

void SendLastImageJob::sendImageToServer()
{
    delete _buffer;
    _buffer = new QBuffer(&_encodedImageData);
    _buffer->open(QIODevice::ReadWrite);
    _image->save(_buffer, "PNG");
    _buffer->seek(0);

    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();

    std::stringstream stream;
    stream << "Web: sending last image with size " << _size.x << " x " << _size.y << ": " << _encodedImageData.size()
           << " bytes";
    loggingService->logMessage(Priority::Important, stream.str());

    _webAccess->sendLastImage(_currentSimulationId, _currentToken, _buffer);

    _state = State::ImageToServerSent;
    _isReady = false;
}

void SendLastImageJob::finish()
{
    _webAccess->requestDisconnect(_currentSimulationId, _currentToken);

    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
    loggingService->logMessage(Priority::Important, "Web: disconnected");

    _state = State::Finished;
    _isReady = true;
}

void SendLastImageJob::imageFromGpuReceived()
{
    if (State::ImageFromGpuRequested != _state) {
        return;
    }

    _isReady = true;
}

void SendLastImageJob::serverReceivedImage()
{
    if (State::ImageToServerSent != _state) {
        return;
    }
    auto loggingService = ServiceLocator::getInstance().getService<LoggingService>();
    loggingService->logMessage(Priority::Important, "Web: last image sent");
    _isReady = true;
}
