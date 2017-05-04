#ifndef SYMBOLEDIT_H
#define SYMBOLEDIT_H

#include <QWidget>

#include "model/Definitions.h"

namespace Ui {
class SymbolEdit;
}

class QTableWidgetItem;
class SymbolEdit : public QWidget
{
    Q_OBJECT
    
public:
    explicit SymbolEdit(QWidget *parent = 0);
    ~SymbolEdit();

    void loadSymbols (SymbolTable* symbolTable);

Q_SIGNALS:
    void symbolTableChanged (); //current symbol table can be obtained from MetadataManager

private Q_SLOTS:
    void addSymbolButtonClicked ();
    void delSymbolButtonClicked ();
    void itemSelectionChanged ();
    void itemContentChanged (QTableWidgetItem* item);

private:
    Ui::SymbolEdit *ui;

    SymbolTable* _symbolTable;
};

#endif // SYMBOLEDIT_H
