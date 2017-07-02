#ifndef CELLCOMPUTEREDIT_H
#define CELLCOMPUTEREDIT_H

#include <QWidget>

namespace Ui {
class CellComputerEdit;
}

class CellComputerEdit : public QWidget
{
    Q_OBJECT
    
public:
    explicit CellComputerEdit(QWidget *parent = 0);
    ~CellComputerEdit();

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
    Ui::CellComputerEdit *ui;
    QTimer* _timer;
    bool _expectCellCompilerAnswer;
};

#endif // CELLCOMPUTEREDIT_H
