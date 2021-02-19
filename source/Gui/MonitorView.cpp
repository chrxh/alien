#include <QPaintEvent>

#include "EngineInterface/MonitorData.h"
#include "StringHelper.h"

#include "MonitorView.h"
#include "ui_MonitorView.h"


MonitorView::MonitorView(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::MonitorView)
{
    ui->setupUi(this);
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
    ui->contentLabel->setText(generateString());
}

bool MonitorView::event(QEvent* event)
{
    if( event->type() == QEvent::Close) {
        Q_EMIT closed();
    }
    QWidget::event(event);
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
	text += parStart + colorTextStart + "Clusters:" + StringHelper::ws(12) + colorEnd;
	text += " " + StringHelper::generateFormattedIntString(_model->numClusters, true)+ " " + parEnd;
	text += parStart + colorTextStart + "Cells:" + StringHelper::ws(15) + colorEnd;
	text += " " + StringHelper::generateFormattedIntString(_model->numCells, true) + " " + parEnd;
	text += parStart + colorTextStart + "Particles:" + StringHelper::ws(11) + colorEnd;
	text += " " + StringHelper::generateFormattedIntString(_model->numParticles, true) + " " + parEnd;
	text += parStart + colorTextStart + "Tokens:" + StringHelper::ws(14) + colorEnd;
    text += " " + StringHelper::generateFormattedIntString(_model->numTokens, true) + " " + parEnd;
    text += parStart + colorTextStart + "Active clusters:" + StringHelper::ws(5) + colorEnd;
    text += " " + StringHelper::generateFormattedIntString(_model->numClustersWithTokens, true) + " " + parEnd;
	return text;
}

