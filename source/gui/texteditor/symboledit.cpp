#include <QScrollBar>

#include "model/metadata/SymbolTable.h"
#include "gui/Settings.h"
#include "gui/Settings.h"

#include "symboledit.h"
#include "ui_symboledit.h"

SymbolEdit::SymbolEdit(QWidget *parent)
	: QWidget(parent), ui(new Ui::SymbolEdit)
{
    ui->setupUi(this);

    //set color
    setStyleSheet("background-color: #000000");
    ui->tableWidget->setStyleSheet(TABLE_STYLESHEET);
    ui->tableWidget->verticalScrollBar()->setStyleSheet(SCROLLBAR_STYLESHEET);
    ui->addSymbolButton->setStyleSheet(BUTTON_STYLESHEET);
    ui->delSymbolButton->setStyleSheet(BUTTON_STYLESHEET);
    QPalette p = ui->addSymbolButton->palette();
    p.setColor(QPalette::ButtonText, BUTTON_TEXT_COLOR);
    ui->addSymbolButton->setPalette(p);
    ui->delSymbolButton->setPalette(p);

	//set font
    ui->tableWidget->setFont(GuiSettings::getGlobalFont());

    //set section length
    ui->tableWidget->horizontalHeader()->resizeSection(0, 285);
    ui->tableWidget->horizontalHeader()->resizeSection(1, 55);

    //connections
    connect(ui->addSymbolButton, SIGNAL(clicked()), this, SLOT(addSymbolButtonClicked()));
    connect(ui->delSymbolButton, SIGNAL(clicked()), this, SLOT(delSymbolButtonClicked()));
    connect(ui->tableWidget, SIGNAL(itemSelectionChanged()), this, SLOT(itemSelectionChanged()));
}

SymbolEdit::~SymbolEdit()
{
    delete ui;
}

void SymbolEdit::loadSymbols(SymbolTable* symbolTable)
{
    _symbolTable = symbolTable;

    //disable notification for item changes
    disconnect(ui->tableWidget, SIGNAL(itemChanged(QTableWidgetItem*)), 0, 0);

    //delete rows in the table
    while( ui->tableWidget->rowCount() > 0 )
        ui->tableWidget->removeRow(0);

    //create entries in the table
    QMapIterator< QString, QString > it = _symbolTable->getTableConstRef();
    int i = 0;
    while( it.hasNext() ) {
        it.next();

        //create new row in table
        ui->tableWidget->insertRow(i);
        ui->tableWidget->setVerticalHeaderItem(i, new QTableWidgetItem(""));
        ui->tableWidget->verticalHeader()->setSectionResizeMode(i, QHeaderView::ResizeToContents);

        //set values
        QString key = it.key();
        QString value = it.value();
        ui->tableWidget->setItem(i, 0, new QTableWidgetItem(key));
        ui->tableWidget->setItem(i, 1, new QTableWidgetItem(value));
        ui->tableWidget->item(i, 0)->setTextColor(CELL_EDIT_DATA_COLOR1);
        ui->tableWidget->item(i, 1)->setTextColor(CELL_EDIT_DATA_COLOR1);
        ++i;
    }

    //set del button
    if( ui->tableWidget->rowCount() > 0 )
        ui->delSymbolButton->setEnabled(true);
    else
        ui->delSymbolButton->setEnabled(false);

    //notify item changes now
    connect(ui->tableWidget, SIGNAL(itemChanged(QTableWidgetItem*)), this, SLOT(itemContentChanged(QTableWidgetItem*)));
}

void SymbolEdit::addSymbolButtonClicked ()
{
    disconnect(ui->tableWidget, SIGNAL(itemChanged(QTableWidgetItem*)), 0, 0);

    //create new row in table
    int row = ui->tableWidget->rowCount();
    ui->tableWidget->insertRow(row);
    ui->tableWidget->setVerticalHeaderItem(row, new QTableWidgetItem(""));
    QTableWidgetItem* item = new QTableWidgetItem("");
    ui->tableWidget->setItem(row, 0, item);
    ui->tableWidget->setItem(row, 1, new QTableWidgetItem(""));
    ui->tableWidget->editItem(item);
    ui->tableWidget->setCurrentCell(row,0);

    //activate del button
    ui->delSymbolButton->setEnabled(true);

    //notify item changes now
    connect(ui->tableWidget, SIGNAL(itemChanged(QTableWidgetItem*)), this, SLOT(itemContentChanged(QTableWidgetItem*)));
}

void SymbolEdit::delSymbolButtonClicked ()
{
    while( !ui->tableWidget->selectedItems().isEmpty() ) {
        QList< QTableWidgetItem* > items = ui->tableWidget->selectedItems();
        int row = items.at(0)->row();
        QString key = ui->tableWidget->item(row, 0)->text();
        _symbolTable->delEntry(key);
        ui->tableWidget->removeRow(row);
    }
    if( ui->tableWidget->rowCount() == 0 )
        ui->delSymbolButton->setEnabled(false);
    Q_EMIT symbolTableChanged();
}

void SymbolEdit::itemSelectionChanged ()
{
    //items selected?
    if( ui->tableWidget->selectedItems().empty() ) {
        ui->delSymbolButton->setEnabled(false);
    }
    else {
        ui->delSymbolButton->setEnabled(true);
    }
}

void SymbolEdit::itemContentChanged (QTableWidgetItem* item)
{
    //clear and update complete symbol table
    _symbolTable->clearTable();
    for(int i = 0; i < ui->tableWidget->rowCount(); ++i) {

        //obtain key and value text
        QString key = ui->tableWidget->item(i, 0)->text();
        QString value = ui->tableWidget->item(i, 1)->text();
        _symbolTable->addEntry(key, value);
    }

    Q_EMIT symbolTableChanged();
}



