#ifndef ADDRECTSTRUCTUREDIALOG_H
#define ADDRECTSTRUCTUREDIALOG_H

#include <QDialog>

namespace Ui {
class AddRectStructureDialog;
}

class AddRectStructureDialog : public QDialog
{
    Q_OBJECT
    
public:
    explicit AddRectStructureDialog(QWidget *parent = 0);
    ~AddRectStructureDialog();

    int getBlockSizeX ();
    int getBlockSizeY ();
    qreal getDistance ();
    qreal getInternalEnergy ();

private:
    Ui::AddRectStructureDialog *ui;
};

#endif // ADDRECTSTRUCTUREDIALOG_H
