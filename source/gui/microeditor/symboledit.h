#ifndef SYMBOLEDIT_H
#define SYMBOLEDIT_H

#include "../../model/metadatamanager.h"

#include <QWidget>

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

    void loadSymbols (MetadataManager* meta);

signals:
    void symbolTableChanged (); //current symbol table can be obtained from MetadataManager

private slots:
    void addSymbolButtonClicked ();
    void delSymbolButtonClicked ();
    void itemSelectionChanged ();
    void itemContentChanged (QTableWidgetItem* item);

private:
    Ui::SymbolEdit *ui;

    MetadataManager* _meta;
};

#endif // SYMBOLEDIT_H
