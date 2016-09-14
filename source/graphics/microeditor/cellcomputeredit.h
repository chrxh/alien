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

    void updateComputerMemory (const QVector< quint8 >& data);
    void updateComputerCode (QString code);
    QString getComputerCode ();

    void setCompilationState (bool error, int line);
    void expectCellCompilerAnswer ();

signals:
    void changesFromComputerMemoryEditor (QVector< quint8 > data);
    void compileButtonClicked (QString code);

private slots:
    void compileButtonClicked_Slot ();
    void timerTimeout ();

    
private:
    Ui::CellComputerEdit *ui;
    QTimer* _timer;
    bool _expectCellCompilerAnswer;
};

#endif // CELLCOMPUTEREDIT_H
