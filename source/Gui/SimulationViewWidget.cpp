#include <QScrollBar>
#include <QTimer>
#include <QGraphicsItem>
#include <QGraphicsBlurEffect>
#include <QFile>
#include <QTextStream>

#include "Gui/Settings.h"
#include "ModelBasic/SimulationAccess.h"
#include "ModelBasic/SimulationController.h"
#include "ModelBasic/SimulationContext.h"
#include "ModelBasic/SpaceProperties.h"

#include "PixelUniverseView.h"
#include "VectorUniverseView.h"
#include "ItemUniverseView.h"

#include "SimulationViewWidget.h"
#include "ui_SimulationViewWidget.h"


SimulationViewWidget::SimulationViewWidget(QWidget *parent)
	: QWidget(parent)
	, ui(new Ui::SimulationViewWidget)
{
    ui->setupUi(this);

    _pixelUniverse = new PixelUniverseView(ui->simulationView, this);
    _vectorUniverse = new VectorUniverseView(ui->simulationView, this);
    _itemUniverse = new ItemUniverseView(ui->simulationView, this);

    ui->simulationView->horizontalScrollBar()->setStyleSheet(Const::ScrollbarStyleSheet);
    ui->simulationView->verticalScrollBar()->setStyleSheet(Const::ScrollbarStyleSheet);

    setStartupScene();
}

SimulationViewWidget::~SimulationViewWidget()
{
    delete ui;
}

void SimulationViewWidget::init(Notifier* notifier, SimulationController* controller, SimulationAccess* access, DataRepository* repository)
{
    auto const InitialZoomFactor = 4.0;

	_controller = controller;

    _pixelUniverse->init(notifier, controller, access, repository);
    _vectorUniverse->init(notifier, controller, access, repository);
    _itemUniverse->init(notifier, controller, repository);

    _vectorUniverse->activate(InitialZoomFactor);

    auto const size = _controller->getContext()->getSpaceProperties()->getSize();
    _vectorUniverse->centerTo({ static_cast<float>(size.x) / 2, static_cast<float>(size.y) / 2 });

    _vectorUniverse->connectView();
    _vectorUniverse->refresh();

    Q_EMIT zoomFactorChanged(InitialZoomFactor);
}

void SimulationViewWidget::connectView()
{
    getActiveUniverseView()->connectView();
}

void SimulationViewWidget::disconnectView()
{
    getActiveUniverseView()->disconnectView();
}

void SimulationViewWidget::refresh()
{
    getActiveUniverseView()->refresh();
}

ActiveView SimulationViewWidget::getActiveView() const
{
    if (_pixelUniverse->isActivated()) {
        return ActiveView::PixelScene;
    }
    if (_vectorUniverse->isActivated()) {
        return ActiveView::VectorScene;
    }
    if (_itemUniverse->isActivated()) {
        return ActiveView::ItemScene;
    }

    THROW_NOT_IMPLEMENTED();
}

void SimulationViewWidget::setActiveScene (ActiveView activeScene)
{
    auto scrollPosX = ui->simulationView->horizontalScrollBar()->value();
    auto scrollPosY = ui->simulationView->verticalScrollBar()->value();

    auto zoom = getZoomFactor();

    auto view = getView(activeScene);
    view->activate(zoom);

    ui->simulationView->horizontalScrollBar()->setValue(scrollPosX); //workaround since UniverseView::centerTo has bad precision
    ui->simulationView->verticalScrollBar()->setValue(scrollPosY);
}

double SimulationViewWidget::getZoomFactor()
{
    return getActiveUniverseView()->getZoomFactor();
}

void SimulationViewWidget::setZoomFactor(double factor)
{
    auto activeView = getActiveUniverseView();
    auto screenCenterPos = activeView->getCenterPositionOfScreen();
    activeView->setZoomFactor(factor);
    activeView->centerTo(screenCenterPos);

    Q_EMIT zoomFactorChanged(factor);
}

QVector2D SimulationViewWidget::getViewCenterWithIncrement ()
{
	auto screenCenterPos = getActiveUniverseView()->getCenterPositionOfScreen();

    QVector2D posIncrement(_posIncrement, -_posIncrement);
    _posIncrement = _posIncrement + 1.0;
    if (_posIncrement > 9.0) {
        _posIncrement = 0.0;
    }
    return screenCenterPos + posIncrement;
}

void SimulationViewWidget::toggleCenterSelection(bool value)
{
    auto activeUniverseView = getActiveUniverseView();
    auto itemUniverseView = dynamic_cast<ItemUniverseView*>(activeUniverseView);
    CHECK(itemUniverseView);

    itemUniverseView->toggleCenterSelection(value);
}

void SimulationViewWidget::setStartupScene()
{
    auto startupScene = new QGraphicsScene(this);
    startupScene->setBackgroundBrush(QBrush(Const::UniverseColor));

    QPixmap startScreenPixmap("://logo.png");
    startupScene->addPixmap(startScreenPixmap);

    QFile file("://Version.txt");
    if (!file.open(QIODevice::ReadOnly)) {
        THROW_NOT_IMPLEMENTED();
    }
    QTextStream in(&file);
    auto version = in.readLine();

    QGraphicsSimpleTextItem* versionTextItem = new QGraphicsSimpleTextItem("Version " + version);
    auto font = versionTextItem->font();
    font.setPixelSize(15);
    versionTextItem->setFont(font);
    versionTextItem->setPos(440, 480);
    versionTextItem->setBrush(Const::StartupTextColor);
    startupScene->addItem(versionTextItem);

    ui->simulationView->setScene(startupScene);
}

UniverseView * SimulationViewWidget::getActiveUniverseView() const
{
    if (_pixelUniverse->isActivated()) {
        return _pixelUniverse;
    }
    if (_vectorUniverse->isActivated()) {
        return _vectorUniverse;
    }
    if (_itemUniverse->isActivated()) {
        return _itemUniverse;
    }

    THROW_NOT_IMPLEMENTED();
}

UniverseView * SimulationViewWidget::getView(ActiveView activeView) const
{
    if (ActiveView::PixelScene == activeView) {
        return _pixelUniverse;
    }
    if (ActiveView::VectorScene == activeView) {
        return _vectorUniverse;
    }
    if (ActiveView::ItemScene== activeView) {
        return _itemUniverse;
    }

    THROW_NOT_IMPLEMENTED();
}



