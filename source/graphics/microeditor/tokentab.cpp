#include "tokentab.h"
#include "ui_tokentab.h"

#include "hexedit.h"
#include "../../globaldata/metadatamanager.h"

#include "../../globaldata/editorsettings.h"
#include "../../globaldata/guisettings.h"
#include "../../globaldata/globalfunctions.h"

#include <QScrollBar>
#include <QSignalMapper>

TokenTab::TokenTab(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::TokenTab),
    _signalMapper(new QSignalMapper(this)),
    _signalMapper2(new QSignalMapper(this)),
    _signalMapper3(new QSignalMapper(this))
{
    ui->setupUi(this);

    //set color
    QPalette p = ui->tokenMemoryLabel->palette();
    p.setColor(QPalette::WindowText, CELL_EDIT_CAPTION_COLOR1);
    ui->tokenMemoryLabel->setPalette(p);
    p.setColor(QPalette::WindowText, CELL_EDIT_DATA_COLOR1);
    p.setColor(QPalette::Text, CELL_EDIT_DATA_COLOR1);
    p.setColor(QPalette::Base, QColor(0,0,0));
    p.setColor(QPalette::Window, QColor(0,0,0));
    ui->tableWidget->setPalette(p);
    ui->tableWidget->setStyleSheet("background-color: #000000; color: #B0B0B0; gridline-color: #303030;");
    ui->tableWidget->verticalScrollBar()->setStyleSheet(SCROLLBAR_STYLESHEET);

//    ui->tableWidget->horizontalHeader()->setPalette(p);
/*    QString s = "::section{background: #000000; font-family: Courier New; font-weight: bold; font-size: 12px;} ";
    ui->tableWidget->horizontalHeader()->setStyleSheet(s);
    ui->tableWidget->verticalHeader()->setStyleSheet(s);*/

    //set font
    ui->tableWidget->setFont(GlobalFunctions::getGlobalFont());

    //set section length
    ui->tableWidget->horizontalHeader()->resizeSection(0, 60);
    ui->tableWidget->horizontalHeader()->resizeSection(1, 195);
    ui->tableWidget->horizontalHeader()->resizeSection(2, 100);
    ui->tableWidget->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);


    connect(ui->tokenEditor, SIGNAL(dataChanged(qreal)), this, SIGNAL(tokenPropChanged(qreal)));
    connect(_signalMapper, SIGNAL(mapped(int)), this, SLOT(tokenMemoryChanged_Slot(int)));
    connect(_signalMapper2, SIGNAL(mapped(int)), this, SLOT(tokenMemoryCursorReachedBeginning_Slot(int)));
    connect(_signalMapper3, SIGNAL(mapped(int)), this, SLOT(tokenMemoryCursorReachedEnd_Slot(int)));
}

TokenTab::~TokenTab()
{
    delete ui;
}

void TokenTab::update (qreal tokenEnergy, const QVector< quint8 >& tokenMemory)
{
//    ui->tokenMemoryEditor->update(tokenMemory);
    ui->tokenEditor->update(tokenEnergy);
    _tokenMemory = tokenMemory;

    //delete rows in the table
    while( ui->tableWidget->rowCount() > 0 )
        ui->tableWidget->removeRow(0);

    //find all addresses from variables
    _hexEditList.clear();
    QMap< quint8, QStringList > addressVarMap;
    QMapIterator< QString, QString > symTblIt(MetadataManager::getGlobalInstance().getSymbolTable());
    while(symTblIt.hasNext()) {
        symTblIt.next();
        QString k = symTblIt.key();
        QString v = symTblIt.value();

        //fast check if variable or not
        if( v.size() > 1 ) {
            if( (v.at(0) == QChar('[')) && (v.at(1) != QChar('['))) {
                int i = v.indexOf("]");
                if( i >= 0) {
                    bool ok(true);
                    quint32 addr = v.mid(1, i-1).toUInt(&ok) % 256;
                    if( ok ) {
                        addressVarMap[addr] << k;
                    }
                }
            }
        }
    }

    int row = 0;
    int tokenMemPointer = 0;
    QMapIterator< quint8, QStringList > addressVarMapIt(addressVarMap);
    do {

        //add address block for variable
        qint32 k = -1;
        if( addressVarMapIt.hasNext() ) {
            addressVarMapIt.next();
            k = addressVarMapIt.key();
            QStringList v = addressVarMapIt.value();
            ui->tableWidget->insertRow(row);

            //set address and var info
            ui->tableWidget->setItem(row, 0, new QTableWidgetItem(QString("%1").arg(k, 2, 16, QChar('0')).toUpper()));
            ui->tableWidget->setItem(row, 1, new QTableWidgetItem(v.join(QChar('\n'))));
            ui->tableWidget->item(row, 0)->setFlags(Qt::NoItemFlags);
            ui->tableWidget->item(row, 0)->setTextAlignment(Qt::AlignTop);
            ui->tableWidget->item(row, 0)->setTextColor(CELL_EDIT_TEXT_COLOR1);
            ui->tableWidget->item(row, 1)->setFlags(Qt::NoItemFlags);
            ui->tableWidget->item(row, 1)->setTextAlignment(Qt::AlignTop);
            ui->tableWidget->item(row, 1)->setTextColor(CELL_EDIT_TEXT_COLOR2);
            ui->tableWidget->setVerticalHeaderItem(row, new QTableWidgetItem(""));

            //create hex editor
            HexEdit* hex = new HexEdit();
            hex->setMinimumHeight(26+15*(v.size()-1));
            hex->setGeometry(0, 0, 100, 26+15*(v.size()-1));
            hex->setMaximumHeight(26+15*(v.size()-1));
            hex->setStyleSheet("border: 0px");
            hex->update(tokenMemory.mid(tokenMemPointer, 1));
            ui->tableWidget->setCellWidget(row, 2, hex);
            ui->tableWidget->verticalHeader()->setSectionResizeMode(row, QHeaderView::ResizeToContents);

            //update signal mapper
            _signalMapper->setMapping(hex, tokenMemPointer);
            _signalMapper2->setMapping(hex, tokenMemPointer);
            _signalMapper3->setMapping(hex, tokenMemPointer);
            connect(hex, SIGNAL(dataChanged(QVector< quint8 >)), _signalMapper, SLOT(map()));
            connect(hex, SIGNAL(cursorReachedBeginning(int)), _signalMapper2, SLOT(map()));
            connect(hex, SIGNAL(cursorReachedEnd(int)), _signalMapper3, SLOT(map()));
            _hexEditList[tokenMemPointer] = hex;

            //update pointer
            ++row;
            ++tokenMemPointer;
        }

        //read next address for variable
        qint32 kNew = -1;
        if( addressVarMapIt.hasNext() ) {
            addressVarMapIt.next();
            kNew = addressVarMapIt.key();
            addressVarMapIt.previous();
        }

        //calc data block address range
        if( k == -1 )
            k = 0;
        else
            k++;
        if( kNew == -1 )
            kNew = 256;

        //add data address block?
        if( kNew > k ) {
            ui->tableWidget->insertRow(row);
            if( kNew > (k+1) )
                ui->tableWidget->setItem(row, 0, new QTableWidgetItem(QString("%1 - %2").arg(k, 2, 16, QChar('0')).arg(kNew, 2, 16, QChar('0')).toUpper()));
            else
                ui->tableWidget->setItem(row, 0, new QTableWidgetItem(QString("%1").arg(k, 2, 16, QChar('0')).toUpper()));
            ui->tableWidget->setItem(row, 1, new QTableWidgetItem("(pure data block)"));
//            ui->tableWidget->setItem(row, 2, new QTableWidgetItem("blabla"));
            ui->tableWidget->item(row, 0)->setFlags(Qt::NoItemFlags);
            ui->tableWidget->item(row, 0)->setTextAlignment(Qt::AlignTop);
            ui->tableWidget->item(row, 0)->setTextColor(CELL_EDIT_TEXT_COLOR1);
            ui->tableWidget->item(row, 1)->setFlags(Qt::NoItemFlags);
            ui->tableWidget->item(row, 1)->setTextAlignment(Qt::AlignTop);
            ui->tableWidget->item(row, 1)->setTextColor(CELL_EDIT_TEXT_COLOR2);
            ui->tableWidget->setVerticalHeaderItem(row, new QTableWidgetItem(""));

            HexEdit* hex = new HexEdit();
            int size = (kNew-k)/6;
            hex->setMaximumHeight(26+13*size);
            hex->setMinimumHeight(26+13*size);
            hex->setGeometry(0, 0, 100, 26+13*size);
            hex->setStyleSheet("border: 0px");
            hex->update(tokenMemory.mid(tokenMemPointer, kNew-k));
            ui->tableWidget->setCellWidget(row, 2, hex);
            ui->tableWidget->verticalHeader()->setSectionResizeMode(row, QHeaderView::ResizeToContents);

            //update signal mapper
            _signalMapper->setMapping(hex, tokenMemPointer);
            _signalMapper2->setMapping(hex, tokenMemPointer);
            _signalMapper3->setMapping(hex, tokenMemPointer);
            connect(hex, SIGNAL(dataChanged(QVector< quint8 >)), _signalMapper, SLOT(map()));
            connect(hex, SIGNAL(cursorReachedBeginning(int)), _signalMapper2, SLOT(map()));
            connect(hex, SIGNAL(cursorReachedEnd(int)), _signalMapper3, SLOT(map()));
            _hexEditList[tokenMemPointer] = hex;

            //update pointer
            ++row;
            tokenMemPointer = tokenMemPointer + kNew-k;
        }
    } while(addressVarMapIt.hasNext());
}

void TokenTab::requestUpdate ()
{
    ui->tokenEditor->requestUpdate();
    emit tokenMemoryChanged(_tokenMemory);
}

void TokenTab::tokenMemoryChanged_Slot (int tokenMemPointer)
{
    HexEdit* hex = _hexEditList[tokenMemPointer];
    if( hex ) {
        QVector< quint8 > newData = hex->getData();
        for(int i = 0; i < newData.size(); ++i) {
            _tokenMemory[tokenMemPointer+i] = newData[i];
        }
    }
    emit tokenMemoryChanged(_tokenMemory);
}

void TokenTab::tokenMemoryCursorReachedBeginning_Slot (int tokenMemPointer)
{
    if(tokenMemPointer > 0) {

        //get cursor column position
//        HexEdit* hex = _hexEditList[tokenMemPointer];
//        int oldPos = hex->textCursor().columnNumber();

        //set focus
        HexEdit* preHex = _hexEditList.lowerBound(tokenMemPointer).operator --().value();
        preHex->setFocus(Qt::OtherFocusReason);

        //calc row (QTableWidget does not give correct information because the widget inside an table item is focused...)
        QMap<quint8, HexEdit*>::iterator it(_hexEditList.lowerBound(tokenMemPointer));
        int row = 0;
        while(it.key() != 0) {
            it--;
            row++;
        }

        //set cursor position
//        preHex->textCursor().setPosition(1);//oldPos);

        //scroll to row
        ui->tableWidget->scrollToItem(ui->tableWidget->item(row-1, 0));
    }
}

void TokenTab::tokenMemoryCursorReachedEnd_Slot (int tokenMemPointer)
{
    if(_hexEditList.lowerBound(tokenMemPointer).operator ++() != _hexEditList.end()) {

        //set focus
        HexEdit* nextHex = _hexEditList.lowerBound(tokenMemPointer).operator ++().value();
        nextHex->setFocus(Qt::OtherFocusReason);

        //calc row (QTableWidget does not give correct information because the widget inside an table item is focused...)
        QMap<quint8, HexEdit*>::iterator it(_hexEditList.lowerBound(tokenMemPointer));
        int row = 0;
        while(it.key() != 0) {
            it--;
            row++;
        }

        //scroll to row
        ui->tableWidget->scrollToItem(ui->tableWidget->item(row+1, 0));
    }
}


