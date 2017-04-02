#ifndef SYMBOLEDIT_H
#define SYMBOLEDIT_H

#include <QWidget>

#include "model/definitions.h"

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

signals:
    void symbolTableChanged (); //current symbol table can be obtained from MetadataManager

private slots:
    void addSymbolButtonClicked ();
    void delSymbolButtonClicked ();
    void itemSelectionChanged ();
    void itemContentChanged (QTableWidgetItem* item);

private:
    Ui::SymbolEdit *ui;

    SymbolTable* _symbolTable;
};

#endif // SYMBOLEDIT_H
