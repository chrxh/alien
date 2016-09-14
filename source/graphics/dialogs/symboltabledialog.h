#ifndef SYMBOLTABLEDIALOG_H
#define SYMBOLTABLEDIALOG_H

//#include "../../globaldata/metadatamanager.h"

#include <QDialog>

namespace Ui {
class SymbolTableDialog;
}

class MetaDataManager;
class SymbolTableDialog : public QDialog
{
    Q_OBJECT
    
public:
    explicit SymbolTableDialog(MetaDataManager* meta, QWidget *parent = 0);
    ~SymbolTableDialog();

    void updateSymbolTable (MetaDataManager* meta);

private slots:
    void setSymbolTableToWidget (MetaDataManager* meta);
    void itemSelectionChanged ();
    void addButtonClicked ();
    void delButtonClicked ();
    void defaultButtonClicked ();
    void loadButtonClicked ();
    void saveButtonClicked ();
    void mergeWithButtonClicked ();

private:
    Ui::SymbolTableDialog *ui;
};

#endif // SYMBOLTABLEDIALOG_H
