#pragma once

#include <QObject>

#include "Web/Definitions.h"

#include "Definitions.h"

class StartupController : public QObject
{
    Q_OBJECT

public:
    StartupController(WebAccess* access, QWidget* parent);
    virtual ~StartupController() = default;

    void init();

    QWidget* getWidget() const;

private:
    void initWidget();

    Q_SLOT void fadeout();
    Q_SLOT void currentVersionReceived(string currentVersion);

    QWidget* _parent = nullptr;
    WebAccess* _access = nullptr;
    QGraphicsView* _graphicsView = nullptr;
    QGraphicsScene* _scene = nullptr;
    QGraphicsSimpleTextItem* _newVersionTextItem = nullptr;
    QString _thisVersion;

    QTimer* _timer = nullptr;
    int _fadeoutProgress = 0;
};
