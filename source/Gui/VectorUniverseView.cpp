#include <QGraphicsPixmapItem>
#include <QGraphicsSceneMouseEvent>
#include <QtCore/qmath.h>
#include <QMatrix4x4>

#include "Base/ServiceLocator.h"
#include "ModelBasic/PhysicalActions.h"
#include "ModelBasic/ModelBasicBuilderFacade.h"
#include "ModelBasic/SimulationAccess.h"
#include "ModelBasic/SimulationController.h"
#include "ModelBasic/SimulationContext.h"
#include "ModelBasic/SpaceProperties.h"
#include "Gui/ViewportInterface.h"
#include "Gui/Settings.h"
#include "Gui/Notifier.h"

#include "CoordinateSystem.h"
#include "DataRepository.h"
#include "VectorImageSectionItem.h"
#include "VectorUniverseView.h"

VectorUniverseView::VectorUniverseView(QObject* parent)
{
    setBackgroundBrush(QBrush(Const::BackgroundColor));
    update();
}

VectorUniverseView::~VectorUniverseView()
{
}

void VectorUniverseView::init(
    Notifier* notifier,
    SimulationController* controller,
    SimulationAccess* access,
    DataRepository* repository,
    ViewportInterface* viewport)
{
    _controller = controller;
    _viewport = viewport;
    _repository = repository;
    _notifier = notifier;

    SET_CHILD(_access, access);

    delete _imageSectionItem;
    auto const viewportRect = _viewport->getRect();

    IntVector2D size = _controller->getContext()->getSpaceProperties()->getSize();
    _imageSectionItem = new VectorImageSectionItem(_viewport, QRectF(0, 0, size.x, size.y), 8, repository->getImageMutex());

    addItem(_imageSectionItem);

    QGraphicsScene::setSceneRect(0, 0, size.x * 8, size.y * 8);

    update();
}
#include <iostream>

void VectorUniverseView::activate()
{
    deactivate();
    _connections.push_back(connect(_controller, &SimulationController::nextFrameCalculated, this, &VectorUniverseView::requestImage));
    _connections.push_back(connect(_notifier, &Notifier::notifyDataRepositoryChanged, this, &VectorUniverseView::receivedNotifications));
    _connections.push_back(connect(_repository, &DataRepository::imageReady, this, &VectorUniverseView::imageReady, Qt::QueuedConnection));
    _connections.push_back(connect(_viewport, &ViewportInterface::scrolled, this, &VectorUniverseView::scrolled));

    IntVector2D size = _controller->getContext()->getSpaceProperties()->getSize();
    auto image = _imageSectionItem->getImageOfVisibleRect();
    _repository->requireVectorImageFromSimulation(
        { { 0, 0 },{ image->width() / 8, image->height()/ 8 } }, image);
    _isActived = true;
}

void VectorUniverseView::deactivate()
{
    _isActived = false;
    for (auto const& connection : _connections) {
        disconnect(connection);
    }
    _connections.clear();
}

void VectorUniverseView::refresh()
{
    requestImage();
}

void VectorUniverseView::mousePressEvent(QGraphicsSceneMouseEvent * event)
{
    if (!_controller->getRun()) {
        QVector2D pos(event->scenePos().x(), event->scenePos().y());
        _access->selectEntities(pos);
        requestImage();
    }
}

void VectorUniverseView::mouseMoveEvent(QGraphicsSceneMouseEvent * e)
{
    auto const pos = QVector2D(e->scenePos().x(), e->scenePos().y());
    auto const lastPos = QVector2D(e->lastScenePos().x(), e->lastScenePos().y());

    if (_controller->getRun()) {
        if (e->buttons() == Qt::MouseButton::LeftButton) {
            auto const force = (pos - lastPos) / 10;
            _access->applyAction(boost::make_shared<_ApplyForceAction>(lastPos, pos, force));
        }
        if (e->buttons() == Qt::MouseButton::RightButton) {
            auto const force = (pos - lastPos) / 10;
            _access->applyAction(boost::make_shared<_ApplyRotationAction>(lastPos, pos, force));
        }
    }
    else {
        if (e->buttons() == Qt::MouseButton::LeftButton) {
            auto const displacement = pos - lastPos;
            _access->applyAction(boost::make_shared<_MoveSelectionAction>(displacement));
            requestImage();
        }
    }
}

void VectorUniverseView::mouseReleaseEvent(QGraphicsSceneMouseEvent * event)
{
    if (!_controller->getRun()) {
        _access->deselectAll();
        requestImage();
    }
}

void VectorUniverseView::receivedNotifications(set<Receiver> const & targets)
{
    if (targets.find(Receiver::VisualEditor) == targets.end()) {
        return;
    }

    requestImage();
}

void VectorUniverseView::requestImage()
{
    if (_isActived) {
        std::cerr << "request image" << std::endl;
        IntRect rect = _viewport->getRect();
        _repository->requireVectorImageFromSimulation({ { rect.p1.x, rect.p1.y }, { rect.p1.x + 50, rect.p1.y + 50 } }, _imageSectionItem->getImageOfVisibleRect());
    }
}

void VectorUniverseView::imageReady()
{
    update();
}

void VectorUniverseView::scrolled()
{
    requestImage();
}

