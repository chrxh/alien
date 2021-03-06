#include <QScrollBar>
#include <QSignalMapper>

#include "EngineInterface/SymbolTable.h"
#include "Gui/Settings.h"
#include "Gui/Settings.h"

#include "DataEditController.h"
#include "DataEditModel.h"
#include "HexEditWidget.h"
#include "TokenEditTab.h"
#include "ui_TokenEditTab.h"

TokenEditTab::TokenEditTab(QWidget *parent) :
    QWidget(parent),
    ui(new Ui::TokenEditTab),
    _signalMapper(new QSignalMapper(this)),
    _signalMapper2(new QSignalMapper(this)),
    _signalMapper3(new QSignalMapper(this))
{
    ui->setupUi(this);
    ui->tableWidget->setFont(GuiSettings::getGlobalFont());

    //set color
    QPalette p = ui->tokenMemoryLabel->palette();
    p.setColor(QPalette::WindowText, Const::CellEditCaptionColor1);
    ui->tokenMemoryLabel->setPalette(p);
    p.setColor(QPalette::WindowText, Const::CellEditDataColor1);
    p.setColor(QPalette::Text, Const::CellEditDataColor1);
    p.setColor(QPalette::Base, QColor(0,0,0));
    p.setColor(QPalette::Window, QColor(0,0,0));
    ui->tableWidget->setPalette(p);
    ui->tableWidget->setStyleSheet("background-color: #000000; color: #B0B0B0; gridline-color: #303030;");
    ui->tableWidget->verticalScrollBar()->setStyleSheet(Const::ScrollbarStyleSheet);

    //set font
    ui->tableWidget->setFont(GuiSettings::getGlobalFont());

    //set section length
    ui->tableWidget->horizontalHeader()->resizeSection(0, 60);
    ui->tableWidget->horizontalHeader()->resizeSection(1, 195);
    ui->tableWidget->horizontalHeader()->resizeSection(2, 100);
    ui->tableWidget->setVerticalScrollMode(QAbstractItemView::ScrollPerPixel);

    connect(
        _signalMapper,
        &QSignalMapper::mappedInt,
        this,
        &TokenEditTab::tokenMemoryChanged);
    connect(
        _signalMapper2,
        &QSignalMapper::mappedInt,
        this,
        &TokenEditTab::tokenMemoryCursorReachedBeginning);
    connect(
        _signalMapper3,
        &QSignalMapper::mappedInt,
        this,
        &TokenEditTab::tokenMemoryCursorReachedEnd);
}

TokenEditTab::~TokenEditTab()
{
    delete ui;
}

void TokenEditTab::init(DataEditModel * model, DataEditController * controller, int tokenIndex)
{
	_model = model;
	_controller = controller;
	_tokenIndex = tokenIndex;

	ui->tokenEditWidget->init(model, controller, tokenIndex);
}

namespace
{
	boost::optional<int> getMemoryLocationOfSymbol(QString const& str)
	{
		if (str.size() > 1) {
			if ((str.at(0) == QChar('[')) && (str.at(1) != QChar('['))) {
				int i = str.indexOf("]");
				if (i >= 0) {
                    if (str.mid(1, 2) == QString("0x")) {
                        bool ok(true);
                        quint32 addr = str.mid(3, i - 3).toUInt(&ok, 16) % 256;
                        if (ok) {
                            return addr;
                        }
                    }
                    else {
                        bool ok(true);
                        quint32 addr = str.mid(1, i - 1).toUInt(&ok) % 256;
                        if (ok) {
                            return addr;
                        }
                    }
				}
			}
		}
		return boost::none;
	}
}

namespace
{
    int calcHeight(int numRows)
	{
		return 26 + 13 * (numRows - 1);
	}
}

void TokenEditTab::updateDisplay()
{

	auto const& token = _model->getTokenToEditRef(_tokenIndex);
	ui->tokenEditWidget->updateDisplay();

	//delete rows in the table
	while (ui->tableWidget->rowCount() > 0)
		ui->tableWidget->removeRow(0);

	//find all addresses from variables
	_hexEditByStartAddress.clear();
	QMap< quint8, QStringList > addressVarMap;
    auto const entries = _model->getSymbolTable()->getEntries();
	for(auto const& keyAndValue : entries) {
		QString k = QString::fromStdString(keyAndValue.first);
		QString v = QString::fromStdString(keyAndValue.second);

		//fast check if variable or not
		if (boost::optional<int> addr = getMemoryLocationOfSymbol(v)) {
			addressVarMap[*addr] << k;
		}
	}

	int row = 0;
	int tokenMemPointer = 0;
	QMapIterator< quint8, QStringList > addressVarMapIt(addressVarMap);
	do {

		//add address block for variable
		qint32 k = -1;
		if (addressVarMapIt.hasNext()) {
			addressVarMapIt.next();
			k = addressVarMapIt.key();
			QStringList v = addressVarMapIt.value();
			ui->tableWidget->insertRow(row);

			//set address and var info
			ui->tableWidget->setItem(row, 0, new QTableWidgetItem(QString("%1").arg(k, 2, 16, QChar('0')).toUpper()));
			ui->tableWidget->setItem(row, 1, new QTableWidgetItem(v.join(QChar('\n'))));
			ui->tableWidget->item(row, 0)->setFlags(Qt::NoItemFlags);
			ui->tableWidget->item(row, 0)->setTextAlignment(Qt::AlignTop);
			ui->tableWidget->item(row, 0)->setForeground(Const::CellEditTextColor1);
			ui->tableWidget->item(row, 1)->setFlags(Qt::NoItemFlags);
			ui->tableWidget->item(row, 1)->setTextAlignment(Qt::AlignTop);
            ui->tableWidget->item(row, 1)->setForeground(Const::CellEditTextColor2);
			ui->tableWidget->setVerticalHeaderItem(row, new QTableWidgetItem(""));

			//create hex editor
			HexEditWidget* hex = new HexEditWidget();
            auto height = calcHeight(v.size());
            hex->setMinimumHeight(height);
            hex->setGeometry(0, 0, 100, height);
            hex->setMaximumHeight(height);
			hex->setStyleSheet("border: 0px");
			hex->updateDisplay(token.data->mid(tokenMemPointer, 1));
            ui->tableWidget->verticalHeader()->setSectionResizeMode(row, QHeaderView::ResizeToContents);
			ui->tableWidget->setCellWidget(row, 2, hex);

			//update signal mapper
			_signalMapper->setMapping(hex, tokenMemPointer);
			_signalMapper2->setMapping(hex, tokenMemPointer);
			_signalMapper3->setMapping(hex, tokenMemPointer);
			connect(hex, &HexEditWidget::dataChanged, _signalMapper, (void(QSignalMapper::*)()) &QSignalMapper::map);
			connect(hex, &HexEditWidget::cursorReachedBeginning, _signalMapper2, (void(QSignalMapper::*)()) &QSignalMapper::map);
			connect(hex, &HexEditWidget::cursorReachedEnd, _signalMapper3, (void(QSignalMapper::*)())& QSignalMapper::map);
			_hexEditByStartAddress[tokenMemPointer] = hex;

			//update pointer
			++row;
			++tokenMemPointer;
		}

		//read next address for variable
		qint32 kNew = -1;
		if (addressVarMapIt.hasNext()) {
			addressVarMapIt.next();
			kNew = addressVarMapIt.key();
			addressVarMapIt.previous();
		}

		//calc data block address range
		if (k == -1)
			k = 0;
		else
			k++;
		if (kNew == -1)
			kNew = 256;

		//add data address block?
		if (kNew > k) {
			ui->tableWidget->insertRow(row);
			if (kNew > (k + 1))
				ui->tableWidget->setItem(row, 0, new QTableWidgetItem(QString("%1 - %2").arg(k, 2, 16, QChar('0')).arg(kNew, 2, 16, QChar('0')).toUpper()));
			else
				ui->tableWidget->setItem(row, 0, new QTableWidgetItem(QString("%1").arg(k, 2, 16, QChar('0')).toUpper()));
			ui->tableWidget->setItem(row, 1, new QTableWidgetItem("(unnamed data block)"));
			ui->tableWidget->item(row, 0)->setFlags(Qt::NoItemFlags);
			ui->tableWidget->item(row, 0)->setTextAlignment(Qt::AlignTop);
            ui->tableWidget->item(row, 0)->setForeground(Const::CellEditTextColor1);
			ui->tableWidget->item(row, 1)->setFlags(Qt::NoItemFlags);
			ui->tableWidget->item(row, 1)->setTextAlignment(Qt::AlignTop);
            ui->tableWidget->item(row, 1)->setForeground(Const::CellEditTextColor2);
			ui->tableWidget->setVerticalHeaderItem(row, new QTableWidgetItem(""));

			int size = (kNew - k) / 4 + 1;
			HexEditWidget* hex = createHexEditWidget(size, row, tokenMemPointer);
			hex->updateDisplay(token.data->mid(tokenMemPointer, kNew - k));
			_hexEditByStartAddress[tokenMemPointer] = hex;

			//update pointer
			++row;
			tokenMemPointer = tokenMemPointer + kNew - k;
		}
	} while (addressVarMapIt.hasNext());
}

HexEditWidget* TokenEditTab::createHexEditWidget(int size, int row, int tokenMemPointer)
{
	HexEditWidget* hex = new HexEditWidget();
    auto height = calcHeight(size);
    hex->setMaximumHeight(height);
    hex->setMinimumHeight(height);
    hex->setGeometry(0, 0, 100, height);
	hex->setStyleSheet("border: 0px");
	ui->tableWidget->setCellWidget(row, 2, hex);
	ui->tableWidget->verticalHeader()->setSectionResizeMode(row, QHeaderView::ResizeToContents);

	//update signal mapper
	_signalMapper->setMapping(hex, tokenMemPointer);
	_signalMapper2->setMapping(hex, tokenMemPointer);
	_signalMapper3->setMapping(hex, tokenMemPointer);
	connect(hex, &HexEditWidget::dataChanged, _signalMapper, (void(QSignalMapper::*)()) &QSignalMapper::map);
	connect(hex, &HexEditWidget::cursorReachedBeginning, _signalMapper2, (void(QSignalMapper::*)()) &QSignalMapper::map);
	connect(hex, &HexEditWidget::cursorReachedEnd, _signalMapper3, (void(QSignalMapper::*)())& QSignalMapper::map);
	return hex;
}

void TokenEditTab::tokenMemoryChanged (int tokenMemPointer)
{
	auto& token = _model->getTokenToEditRef(_tokenIndex);
	HexEditWidget* hex = _hexEditByStartAddress[tokenMemPointer];
    if( hex ) {
        QByteArray newData = hex->getData();
        for(int i = 0; i < newData.size(); ++i) {
			token.data.get()[tokenMemPointer + i] = newData[i];
        }
    }
	_controller->notificationFromTokenTab();
}

void TokenEditTab::tokenMemoryCursorReachedBeginning (int tokenMemPointer)
{
    if(tokenMemPointer > 0) {

        //get cursor column position
//        HexEdit* hex = _hexEditList[tokenMemPointer];
//        int oldPos = hex->textCursor().columnNumber();

        //set focus
        HexEditWidget* preHex = _hexEditByStartAddress.lowerBound(tokenMemPointer).operator --().value();
        preHex->setFocus(Qt::OtherFocusReason);

        //calc row (QTableWidget does not give correct information because the widget inside an table item is focused...)
        QMap<quint8, HexEditWidget*>::iterator it(_hexEditByStartAddress.lowerBound(tokenMemPointer));
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

void TokenEditTab::tokenMemoryCursorReachedEnd (int tokenMemPointer)
{
    if(_hexEditByStartAddress.lowerBound(tokenMemPointer).operator ++() != _hexEditByStartAddress.end()) {

        //set focus
        HexEditWidget* nextHex = _hexEditByStartAddress.lowerBound(tokenMemPointer).operator ++().value();
        nextHex->setFocus(Qt::OtherFocusReason);

        //calc row (QTableWidget does not give correct information because the widget inside an table item is focused...)
        QMap<quint8, HexEditWidget*>::iterator it(_hexEditByStartAddress.lowerBound(tokenMemPointer));
        int row = 0;
        while(it.key() != 0) {
            it--;
            row++;
        }

        //scroll to row
        ui->tableWidget->scrollToItem(ui->tableWidget->item(row+1, 0));
    }
}


