#include <QPaintEvent>

#include "Gui/StringHelper.h"

#include "MonitorView.h"
#include "MonitorModel.h"
#include "ui_MonitorView.h"


MonitorView::MonitorView(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MonitorView)
{
    ui->setupUi(this);
	ui->infoLabel->setText("number of clusters");
}


MonitorView::~MonitorView()
{
    delete ui;
}

void MonitorView::init(MonitorModel const & model)
{
	_model = model;
	update();
}

void MonitorView::update()
{
	ui->infoLabel->setText(generateString());
}

bool MonitorView::event(QEvent* event)
{
    if( event->type() == QEvent::Close) {
        Q_EMIT closed();
    }
    QMainWindow::event(event);
    return false;
}

QString MonitorView::generateString() const
{
	//define auxiliary strings
	QString parStart = "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">";
	QString parEnd = "</p>";
	QString colorTextStart = "<span style=\"color:" + CELL_EDIT_TEXT_COLOR1.name() + "\">";
	QString colorDataStart = "<span style=\"color:" + CELL_EDIT_DATA_COLOR1.name() + "\">";
	QString colorData2Start = "<span style=\"color:" + CELL_EDIT_DATA_COLOR2.name() + "\">";
	QString colorEnd = "</span>";
	QString text;

	//generate formatted string
	text += parStart + colorTextStart + "number of clusters:" + StringHelper::ws(5) + colorEnd;
	text += " " + StringHelper::generateFormattedIntString(_model->numClusters)+ " " + parEnd;
	text += parStart + colorTextStart + "number of cells:" + StringHelper::ws(8) + colorEnd;
	text += " " + StringHelper::generateFormattedIntString(_model->numCells) + " " + parEnd;
	text += parStart + colorTextStart + "number of particles:" + StringHelper::ws(4) + colorEnd;
	text += " " + StringHelper::generateFormattedIntString(_model->numParticles) + " " + parEnd;
	text += parStart + colorTextStart + "number of tokens:" + StringHelper::ws(7) + colorEnd;
	text += " " + StringHelper::generateFormattedIntString(_model->numTokens) + " " + parEnd;
	text += parStart + colorTextStart + "total internal energy:" + StringHelper::ws(2) + colorEnd;
	text += " " + StringHelper::generateFormattedRealString(_model->totalInternalEnergy) + " " + parEnd;
	text += parStart + colorTextStart + "total kinetic energy:" + StringHelper::ws(3) + colorEnd;
	text += " " + StringHelper::generateFormattedRealString(_model->totalKineticEnergy) + " " + parEnd;
	text += parStart + colorTextStart + "&#47;linear part:" + StringHelper::ws(11) + colorEnd;
	text += " " + StringHelper::generateFormattedRealString(_model->totalKineticEnergyTranslationalPart) + " " + parEnd;
	text += parStart + colorTextStart + "&#47;rotational part:" + StringHelper::ws(7) + colorEnd;
	text += " " + StringHelper::generateFormattedRealString(_model->totalKineticEnergyRotationalPart) + " " + parEnd;
	text += parStart + colorTextStart + "overall energy:" + StringHelper::ws(9) + colorEnd;
	text += " " + StringHelper::generateFormattedRealString(_model->totalInternalEnergy + _model->totalKineticEnergy) + " " + parEnd;
	return text;
}

