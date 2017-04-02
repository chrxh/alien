#ifndef SYMBOLTABLEDIALOG_H
#define SYMBOLTABLEDIALOG_H

#include <QDialog>

#include "model/definitions.h"
#include "model/metadata/symboltable.h"

namespace Ui {
class SymbolTableDialog;
}

class SymbolTableDialog : public QDialog
{
    Q_OBJECT
    
public:
	SymbolTableDialog(SymbolTable* symbolTable, QWidget *parent = 0);
	~SymbolTableDialog();

    SymbolTable const& getNewSymbolTableRef ();

private slots:
    void symbolTableToWidgets ();
    void itemSelectionChanged ();
    void addButtonClicked ();
    void delButtonClicked ();
    void defaultButtonClicked ();
    void loadButtonClicked ();
    void saveButtonClicked ();
    void mergeWithButtonClicked ();

private:
	void widgetsToSymbolTable();

    Ui::SymbolTableDialog *ui;
	SymbolTable _symbolTable;
};

#endif // SYMBOLTABLEDIALOG_H
