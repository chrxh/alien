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
    _timer->start(std::chrono::milliseconds(1000));
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
    versionTextItem->setPos(300, 330);
    versionTextItem->setBrush(Const::StartupTextColor);
    _scene->addItem(versionTextItem);

    _newVersionTextItem = new QGraphicsSimpleTextItem(" ");
    _newVersionTextItem->setFont(font);
    _newVersionTextItem->setPos(240, 355);
    _newVersionTextItem->setBrush(Const::StartupNewVersionTextColor);
    _scene->addItem(_newVersionTextItem);

    _graphicsView = new QGraphicsView();
    _graphicsView->setScene(_scene);
    _graphicsView->setGeometry(0, 0, 800, 400);
}

void StartupController::fadeout()
{
    if (0 == _fadeoutProgress) {
        _timer->setSingleShot(false);
        _timer->start(std::chrono::milliseconds(30));
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

namespace
{
    bool parseVersion(std::string const& version, int& major, int& minor, int& patch)
    {
        auto versionQString = QString::fromStdString(version);
        auto versionFragments = versionQString.split(QLatin1Char('.'));
        if (versionFragments.size() != 3) {
            return false;
        }
        auto majorStr = versionFragments[0];
        auto minorStr = versionFragments[1];
        auto patchStr = versionFragments[2];

        bool success;
        major = majorStr.toInt(&success);
        if (!success) {
            return false;
        }
        minor = minorStr.toInt(&success);
        if (!success) {
            return false;
        }
        patch = patchStr.toInt(&success);
        if (!success) {
            return false;
        }
        return true;
    }
}

void StartupController::currentVersionReceived(string currentVersion)
{
    auto currentVersionQString = QString::fromStdString(currentVersion);
    if (_thisVersion != currentVersionQString && _newVersionTextItem) {

        int major;
        int minor;
        int patch;
        if (parseVersion(currentVersion, major, minor, patch)) {
            _newVersionTextItem->setText(QString("- newer version %1 available -").arg(currentVersionQString));

            _fadeoutProgress = 0;
            _timer->setSingleShot(true);
            _timer->start(std::chrono::milliseconds(1000));
        }
    }
}
