#include <QPaintEvent>

#include "ModelBasic/MonitorData.h"
#include "Gui/StringHelper.h"

#include "MonitorView.h"
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

void MonitorView::init(MonitorDataSP const& model)
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
	QString colorTextStart = "<span style=\"color:" + Const::CellEditTextColor1.name() + "\">";
	QString colorDataStart = "<span style=\"color:" + Const::CellEditDataColor1.name() + "\">";
	QString colorData2Start = "<span style=\"color:" + Const::CellEditDataColor2.name() + "\">";
	QString colorEnd = "</span>";
	QString text;

	//generate formatted string
	text += parStart + colorTextStart + "number of clusters:" + StringHelper::ws(12) + colorEnd;
	text += " " + StringHelper::generateFormattedIntString(_model->numClusters, true)+ " " + parEnd;
	text += parStart + colorTextStart + "number of cells:" + StringHelper::ws(15) + colorEnd;
	text += " " + StringHelper::generateFormattedIntString(_model->numCells, true) + " " + parEnd;
	text += parStart + colorTextStart + "number of particles:" + StringHelper::ws(11) + colorEnd;
	text += " " + StringHelper::generateFormattedIntString(_model->numParticles, true) + " " + parEnd;
	text += parStart + colorTextStart + "number of tokens:" + StringHelper::ws(14) + colorEnd;
    text += " " + StringHelper::generateFormattedIntString(_model->numTokens, true) + " " + parEnd;
    text += parStart + colorTextStart + "number of active clusters:" + StringHelper::ws(5) + colorEnd;
    text += " " + StringHelper::generateFormattedIntString(_model->numClustersWithTokens, true) + " " + parEnd;
    text += parStart + colorTextStart + "total internal energy:" + StringHelper::ws(9) + colorEnd;
	text += " " + StringHelper::generateFormattedRealString(_model->totalInternalEnergy, true) + " " + parEnd;
	text += parStart + colorTextStart + "total kinetic energy:" + StringHelper::ws(10) + colorEnd;
	double totalKineticEnergy = _model->totalLinearKineticEnergy + _model->totalRotationalKineticEnergy;
	text += " " + StringHelper::generateFormattedRealString(totalKineticEnergy, true) + " " + parEnd;
	text += parStart + colorTextStart + "&#47;linear part:" + StringHelper::ws(18) + colorEnd;
	text += " " + StringHelper::generateFormattedRealString(_model->totalLinearKineticEnergy, true) + " " + parEnd;
	text += parStart + colorTextStart + "&#47;rotational part:" + StringHelper::ws(14) + colorEnd;
	text += " " + StringHelper::generateFormattedRealString(_model->totalRotationalKineticEnergy, true) + " " + parEnd;
	text += parStart + colorTextStart + "overall energy:" + StringHelper::ws(16) + colorEnd;
	double totalEnergy = _model->totalInternalEnergy + totalKineticEnergy;
	text += " " + StringHelper::generateFormattedRealString(totalEnergy, true) + " " + parEnd;
	return text;
}

