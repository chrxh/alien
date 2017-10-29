#pragma once

#include <QTextEdit>
#include <QVector2D>

#include "Model/Api/Definitions.h"
#include "Model/Api/Descriptions.h"
#include "Gui/Definitions.h"

class ParticleEditTab : public QTextEdit
{
    Q_OBJECT
public:
    ParticleEditTab(QWidget *parent = nullptr);
	virtual ~ParticleEditTab() = default;

	void init(DataEditorModel* model, DataEditorController* controller);
    void updateDisplay();

protected:
    void keyPressEvent (QKeyEvent* e);
    void mousePressEvent(QMouseEvent* e);
    void mouseDoubleClickEvent (QMouseEvent* e);
    void wheelEvent (QWheelEvent* e);

private:
	void updateModelAndNotifyController();

	qreal generateNumberFromFormattedString (QString s);
    QString generateFormattedRealString (QString s);
    QString generateFormattedRealString (qreal r);

	DataEditorModel* _model = nullptr;
	DataEditorController* _controller = nullptr;
};
