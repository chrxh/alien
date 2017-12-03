#pragma once
#include <QDialog>

#include "Model/Api/Definitions.h"
#include "Model/Api/SymbolTable.h"

namespace Ui {
class SymbolTableDialog;
}

class SymbolTableDialog : public QDialog
{
    Q_OBJECT
    
public:
	SymbolTableDialog(SymbolTable* symbolTable, QWidget *parent = 0);
	virtual ~SymbolTableDialog();

    SymbolTable* getNewSymbolTable ();

private Q_SLOTS:
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
	SymbolTable* _symbolTable;
};

