#include "cellcomputeredit.h"
#include "ui_cellcomputeredit.h"

#include "../../globaldata/guisettings.h"
#include "../../globaldata/editorsettings.h"

#include <QTimer>

CellComputerEdit::CellComputerEdit(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::CellComputerEdit),
    _timer(new QTimer(this)),
    _expectCellCompilerAnswer(false)
{
    ui->setupUi(this);

    //set colors
    ui->compileButton2->setStyleSheet(BUTTON_STYLESHEET);

    QPalette p = ui->computerMemoryLabel2->palette();
    p.setColor(QPalette::WindowText, CELL_EDIT_CAPTION_COLOR1);
    ui->computerMemoryLabel2->setPalette(p);
    ui->computerCodeLabel2->setPalette(p);

    //connections
    connect(ui->computerMemoryEditor2, SIGNAL(dataChanged(QVector< quint8 >)), this, SIGNAL(changesFromComputerMemoryEditor(QVector< quint8 >)));
    connect(ui->compileButton2, SIGNAL(clicked()), this, SLOT(compileButtonClicked_Slot()));
    connect(_timer, SIGNAL(timeout()), this, SLOT(timerTimeout()));
}

CellComputerEdit::~CellComputerEdit()
{
    delete ui;
}

void CellComputerEdit::updateComputerMemory (const QVector< quint8 >& data)
{
    ui->computerMemoryEditor2->update(data);
}

void CellComputerEdit::updateComputerCode (QString code)
{
    ui->computerCodeEditor2->update(code);
}

QString CellComputerEdit::getComputerCode ()
{
    return ui->computerCodeEditor2->getCode();
}

void CellComputerEdit::setCompilationState (bool error, int line)
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

void CellComputerEdit::expectCellCompilerAnswer ()
{
    _expectCellCompilerAnswer = true;
}

void CellComputerEdit::compileButtonClicked_Slot ()
{
    emit compileButtonClicked(ui->computerCodeEditor2->getCode());
}

void CellComputerEdit::timerTimeout ()
{
    QPalette p = ui->computerCodeEditor2->palette();
    ui->compilationStateLabel2->setPalette(p);
    ui->compilationStateLabel2->setText("");
    _timer->stop();
}


