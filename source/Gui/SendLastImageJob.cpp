#include "SendLastImageJob.h"

#include <iostream>
#include <QBuffer>
#include <QImage>

#include "ModelBasic/SimulationAccess.h"

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
    std::cerr
        << "[Web] get last image with size "
        << _size.x << " x "
        << _size.y
        << std::endl;

    _image = boost::make_shared<QImage>(_size.x, _size.y, QImage::Format_RGB32);
    auto const rect = IntRect{ _pos, IntVector2D{ _pos.x + _size.x, _pos.y + _size.y } };
    _simAccess->requireImage(rect, _image, _mutex);

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

    std::cerr
        << "[Web] sending last image with size "
        << _size.x << " x "
        << _size.y << ": "
        << _encodedImageData.size() << " bytes"
        << std::endl;

    _webAccess->sendLastImage(_currentSimulationId, _currentToken, _buffer);

    _state = State::ImageToServerSent;
    _isReady = false;
}

void SendLastImageJob::finish()
{
    _webAccess->requestDisconnect(_currentSimulationId, _currentToken);
    std::cerr << "[Web] disconnected" << std::endl;

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
    std::cerr << "[Web] last image sent" << std::endl;
    _isReady = true;
}
