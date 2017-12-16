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
	SymbolTableDialog(SymbolTable* symbolTable, Serializer* serializer, QWidget *parent = 0);
	virtual ~SymbolTableDialog();

    SymbolTable* getSymbolTable ();

private:
	Q_SLOT void updateWidgetsFromSymbolTable ();
	Q_SLOT void itemSelectionChanged ();
	Q_SLOT void addButtonClicked ();
	Q_SLOT void delButtonClicked ();
	Q_SLOT void defaultButtonClicked ();
	Q_SLOT void loadButtonClicked ();
	Q_SLOT void saveButtonClicked ();
	Q_SLOT void mergeWithButtonClicked ();

private:
	void updateSymbolTableFromWidgets();

	SymbolTable* loadSymbolTable(string filename);
	bool saveSymbolTable(string filename, SymbolTable* symbolTable);

    Ui::SymbolTableDialog *ui;
	SymbolTable* _symbolTable = nullptr;
	Serializer* _serializer = nullptr;
};

