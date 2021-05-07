#include <QWidget>
#include <QLabel>
#include <QGraphicsView>
#include <QFile>
#include <QTextStream>
#include <QGraphicsSimpleTextItem>
#include <QGraphicsOpacityEffect>
#include <QTimer>

#include "Web/WebAccess.h"

#include "StartupController.h"
#include "Settings.h"

StartupController::StartupController(WebAccess* access, QWidget* parent)
    : QObject(parent)
    , _parent(parent)
    , _access(access)
{
    connect(access, &WebAccess::currentVersionReceived, this, &StartupController::currentVersionReceived);
 
    initWidget();

    _access->requestCurrentVersion();

    _timer = new QTimer(this);
    connect(_timer, &QTimer::timeout, this, &StartupController::fadeout);
    _timer->setSingleShot(true);
    _timer->start(std::chrono::milliseconds(150));
}

QWidget* StartupController::getWidget() const
{
    return _graphicsView;
}

void StartupController::initWidget()
{
    _scene = new QGraphicsScene(this);
    _scene->setBackgroundBrush(QBrush(Const::UniverseColor));

    QPixmap startScreenPixmap("://logo.png");
    _scene->addPixmap(startScreenPixmap);

    QFile file("://Version.txt");
    if (!file.open(QIODevice::ReadOnly)) {
        THROW_NOT_IMPLEMENTED();
    }
    QTextStream in(&file);
    _thisVersion = in.readLine();

    auto versionTextItem = new QGraphicsSimpleTextItem("Version " + _thisVersion);
    auto font = versionTextItem->font();
    font.setPixelSize(15);
    versionTextItem->setFont(font);
    versionTextItem->setPos(440, 480);
    versionTextItem->setBrush(Const::StartupTextColor);
    _scene->addItem(versionTextItem);

    _newVersionTextItem = new QGraphicsSimpleTextItem(" ");
    _newVersionTextItem->setFont(font);
    _newVersionTextItem->setPos(380, 500);
    _newVersionTextItem->setBrush(Const::StartupNewVersionTextColor);
    _scene->addItem(_newVersionTextItem);

    _graphicsView = new QGraphicsView();
    _graphicsView->setScene(_scene);
    _graphicsView->setGeometry(0, 0, 1000, 570);
}

void StartupController::fadeout()
{
    if (0 == _fadeoutProgress) {
        _timer->setSingleShot(false);
        _timer->start(std::chrono::milliseconds(15));
    }
    if (0 < _fadeoutProgress && _fadeoutProgress < 100) {
        auto transparencyEffect = new QGraphicsOpacityEffect(this);
        transparencyEffect->setOpacity(1.0 - static_cast<double>(_fadeoutProgress) / 100.0);
        _graphicsView->setGraphicsEffect(transparencyEffect);
    }
    if (100 == _fadeoutProgress) {
        delete _graphicsView;
        delete _timer;
        _newVersionTextItem = nullptr;
    }
    ++_fadeoutProgress;
}

void StartupController::currentVersionReceived(string currentVersion)
{
    auto currentVersionQString = QString::fromStdString(currentVersion);
    if (_thisVersion != currentVersionQString && _newVersionTextItem) {

        _newVersionTextItem->setText(QString("(newer version %1 available)").arg(currentVersionQString));

        _fadeoutProgress = 0;
        _timer->setSingleShot(true);
        _timer->start(std::chrono::milliseconds(170));
    }
}
