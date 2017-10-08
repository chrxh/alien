#include "CellComputerEditWidget.h"
#include "ui_CellComputerEditWidget.h"

#include "gui/Settings.h"
#include "gui/Settings.h"

#include <QTimer>

CellComputerEditWidget::CellComputerEditWidget(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::CellComputerEditWidget),
    _timer(new QTimer(this)),
    _expectCellCompilerAnswer(false)
{
    ui->setupUi(this);

    //set colors
    ui->compileButton2->setStyleSheet(BUTTON_STYLESHEET);

    QPalette p = ui->memoryLabel->palette();
    p.setColor(QPalette::WindowText, CELL_EDIT_CAPTION_COLOR1);
    ui->memoryLabel->setPalette(p);
    ui->codeLabel->setPalette(p);

    //connections
    connect(ui->memoryEditor, SIGNAL(dataChanged(QByteArray)), this, SIGNAL(changesFromComputerMemoryEditor(QByteArray)));
    connect(ui->compileButton2, SIGNAL(clicked()), this, SLOT(compileButtonClicked_Slot()));
    connect(_timer, SIGNAL(timeout()), this, SLOT(timerTimeout()));
}

CellComputerEditWidget::~CellComputerEditWidget()
{
    delete ui;
}

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

void CellComputerEditWidget::setCompilationState (bool error, int line)
{
    if( _expectCellCompilerAnswer ) {
        _expectCellCompilerAnswer = false;
        if( error ) {
            QPalette p = ui->compilationStateLabel2->palette();
            p.setColor(QPalette::Window, QColor(0x70,0,0));
            p.setColor(QPalette::WindowText, QColor(0xFF,0,0));
            ui->compilationStateLabel2->setPalette(p);
            ui->compilationStateLabel2->setText(" error at line " + QString::number(line));
        }
        else {
            QPalette p = ui->compilationStateLabel2->palette();
            p.setColor(QPalette::Window, QColor(0,0x70,0));
            p.setColor(QPalette::WindowText, QColor(0,0xFF,0));
            ui->compilationStateLabel2->setPalette(p);
            ui->compilationStateLabel2->setText(" successful");
        }
        _timer->start(2000);
    }
}

void CellComputerEditWidget::expectCellCompilerAnswer ()
{
    _expectCellCompilerAnswer = true;
}

void CellComputerEditWidget::compileButtonClicked_Slot ()
{
    Q_EMIT compileButtonClicked(ui->codeEditWidget->getCode());
}

void CellComputerEditWidget::timerTimeout ()
{
    QPalette p = ui->codeEditWidget->palette();
    ui->compilationStateLabel2->setPalette(p);
    ui->compilationStateLabel2->setText("");
    _timer->stop();
}


