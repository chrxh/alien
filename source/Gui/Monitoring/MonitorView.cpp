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
	//number of clusters
	//number of cells
	//number of particles
	//number of token
	//total internal energy
	//token kinetic energy
	//-> translational part
	//-> rotational part
	//overall energy

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
	text += parStart + colorTextStart + "number of clusters:" + StringHelper::ws(7) + colorEnd;
	text += " " + StringHelper::generateFormattedIntString(_model->numClusters)+ " " + parEnd;

	return text;
}

