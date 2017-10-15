#include <QTimer>

#include "Model/Api/CellComputerCompiler.h"
#include "Gui/Settings.h"
#include "Gui/Settings.h"

#include "ui_CellComputerEditWidget.h"
#include "CodeEditWidget.h"
#include "DataEditorController.h"
#include "DataEditorModel.h"
#include "CellComputerEditWidget.h"

CellComputerEditWidget::CellComputerEditWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::CellComputerEditWidget),
    _timer(new QTimer(this))
{
    ui->setupUi(this);

    //set colors
    ui->compileButton->setStyleSheet(BUTTON_STYLESHEET);

    QPalette p = ui->memoryLabel->palette();
    p.setColor(QPalette::WindowText, CELL_EDIT_CAPTION_COLOR1);
    ui->memoryLabel->setPalette(p);
    ui->codeLabel->setPalette(p);

    //connections
    connect(ui->memoryEditor, SIGNAL(dataChanged(QByteArray)), this, SIGNAL(changesFromComputerMemoryEditor(QByteArray)));
    connect(ui->compileButton, SIGNAL(clicked()), this, SLOT(compileButtonClicked()));
    connect(_timer, SIGNAL(timeout()), this, SLOT(timerTimeout()));
}

CellComputerEditWidget::~CellComputerEditWidget()
{
    delete ui;
}

void CellComputerEditWidget::init(DataEditorModel * model, DataEditorController * controller, CellComputerCompiler * compiler)
{
	_model = model;
	_controller = controller;
	_compiler = compiler;
	ui->codeEditWidget->init(model, controller, compiler);
}

/*
void CellComputerEditWidget::updateComputerMemory(QByteArray const& data)
{
    ui->memoryEditor->update(data);
}

void CellComputerEditWidget::updateComputerCode (QString code)
{
    ui->codeEditWidget->update(code);
}

QString CellComputerEditWidget::getComputerCode ()
{
    return ui->codeEditWidget->getCode();
}
*/

void CellComputerEditWidget::setCompilationState (bool error, int line)
{
    if( error ) {
        QPalette p = ui->compilationStateLabel->palette();
        p.setColor(QPalette::Window, QColor(0x70,0,0));
        p.setColor(QPalette::WindowText, QColor(0xFF,0,0));
        ui->compilationStateLabel->setPalette(p);
        ui->compilationStateLabel->setText(" error at line " + QString::number(line));
    }
    else {
        QPalette p = ui->compilationStateLabel->palette();
        p.setColor(QPalette::Window, QColor(0,0x70,0));
        p.setColor(QPalette::WindowText, QColor(0,0xFF,0));
        ui->compilationStateLabel->setPalette(p);
        ui->compilationStateLabel->setText(" successful");
    }
    _timer->start(2000);
}

void CellComputerEditWidget::compileButtonClicked ()
{
	auto const& code = ui->codeEditWidget->getCode();
	CompilationResult result = _compiler->compileSourceCode(code);
	if (result.compilationOk) {
		auto& cell = _model->getCellToEditRef();
		cell.cellFeature->data = result.compilation;
	}
	setCompilationState(!result.compilationOk, result.lineOfFirstError);
	_controller->notificationFromCellComputerEditWidget();
}

void CellComputerEditWidget::timerTimeout ()
{
    QPalette p = ui->codeEditWidget->palette();
    ui->compilationStateLabel->setPalette(p);
    ui->compilationStateLabel->setText("");
    _timer->stop();
}


