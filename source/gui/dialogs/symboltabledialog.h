#ifndef SYMBOLTABLEDIALOG_H
#define SYMBOLTABLEDIALOG_H

#include <QDialog>

namespace Ui {
class SymbolTableDialog;
}

class MetadataManager;
class SymbolTableDialog : public QDialog
{
    Q_OBJECT
    
public:
    explicit SymbolTableDialog (QWidget *parent = 0);
    ~SymbolTableDialog();

    void updateSymbolTable (MetadataManager* meta);

private slots:
    void setSymbolTableToWidget (MetadataManager* meta);
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
