#ifndef ADDHEXAGONSTRUCTUREDIALOG_H
#define ADDHEXAGONSTRUCTUREDIALOG_H

#include <QDialog>

namespace Ui {
class AddHexagonStructureDialog;
}

class AddHexagonStructureDialog : public QDialog
{
    Q_OBJECT
    
public:
    explicit AddHexagonStructureDialog(QWidget *parent = 0);
    ~AddHexagonStructureDialog();

    int getLayers ();
    qreal getDistance ();
    qreal getInternalEnergy ();

private:
    Ui::AddHexagonStructureDialog *ui;
};

#endif // ADDHEXAGONSTRUCTUREDIALOG_H
