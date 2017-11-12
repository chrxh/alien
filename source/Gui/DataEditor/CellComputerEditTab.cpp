#include <QTimer>

#include "Model/Api/CellComputerCompiler.h"
#include "Gui/Settings.h"
#include "Gui/Settings.h"

#include "ui_CellComputerEditTab.h"
#include "CodeEditWidget.h"
#include "DataEditController.h"
#include "DataEditModel.h"
#include "CellComputerEditTab.h"

CellComputerEditTab::CellComputerEditTab(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::CellComputerEditTab),
    _timer(new QTimer(this))
{
    ui->setupUi(this);

    //set colors
    ui->compileButton->setStyleSheet(BUTTON_STYLESHEET);

    QPalette p = ui->memoryLabel->palette();
    p.setColor(QPalette::WindowText, CELL_EDIT_CAPTION_COLOR1);
    ui->memoryLabel->setPalette(p);
    ui->codeLabel->setPalette(p);

    connect(ui->compileButton, &QToolButton::clicked, this, &CellComputerEditTab::compileButtonClicked);
	connect(_timer, &QTimer::timeout, this, &CellComputerEditTab::timerTimeout);
	connect(ui->memoryEditWidget, &HexEditWidget::dataChanged, this, &CellComputerEditTab::updateFromMemoryEditWidget);
}

CellComputerEditTab::~CellComputerEditTab()
{
    delete ui;
}

void CellComputerEditTab::init(DataEditModel * model, DataEditController * controller, CellComputerCompiler * compiler)
{
	_model = model;
	_controller = controller;
	_compiler = compiler;
	ui->codeEditWidget->init(model, controller, compiler);
}

void CellComputerEditTab::updateDisplay()
{
	ui->codeEditWidget->updateDisplay();
	auto const &cell = _model->getCellToEditRef();
	auto const &data = cell.cellFeature->volatileData;
	ui->memoryEditWidget->updateDisplay(data);
}

void CellComputerEditTab::setCompilationState (bool error, int line)
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

void CellComputerEditTab::compileButtonClicked ()
{
	auto const& code = ui->codeEditWidget->getCode();
	CompilationResult result = _compiler->compileSourceCode(code);
	if (result.compilationOk) {
		auto& cell = _model->getCellToEditRef();
		cell.cellFeature->constData = result.compilation;
		cell.metadata->computerSourcecode = QString::fromStdString(code);
	}
	setCompilationState(!result.compilationOk, result.lineOfFirstError);
	_controller->notificationFromCellComputerTab();
}

void CellComputerEditTab::timerTimeout ()
{
    QPalette p = ui->codeEditWidget->palette();
    ui->compilationStateLabel->setPalette(p);
    ui->compilationStateLabel->setText("");
    _timer->stop();
}

void CellComputerEditTab::updateFromMemoryEditWidget()
{
	auto &cell = _model->getCellToEditRef();
	cell.cellFeature->volatileData = ui->memoryEditWidget->getData();
	_controller->notificationFromCellComputerTab();
}


