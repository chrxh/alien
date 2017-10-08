#pragma once

#include <QWidget>

namespace Ui {
class CellComputerEditWidget;
}

class CellComputerEditWidget : public QWidget
{
    Q_OBJECT
    
public:
    explicit CellComputerEditWidget(QWidget *parent = 0);
    ~CellComputerEditWidget();

    void updateComputerMemory (QByteArray const& data);
    void updateComputerCode (QString code);
    QString getComputerCode ();

    void setCompilationState (bool error, int line);
    void expectCellCompilerAnswer ();

Q_SIGNALS:
    void changesFromComputerMemoryEditor (QByteArray data);
    void compileButtonClicked (QString code);

private Q_SLOTS:
    void compileButtonClicked_Slot ();
    void timerTimeout ();

    
private:
    Ui::CellComputerEditWidget *ui;
    QTimer* _timer;
    bool _expectCellCompilerAnswer;
};
