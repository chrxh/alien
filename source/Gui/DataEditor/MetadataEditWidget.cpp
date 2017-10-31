#include <QMouseEvent>
#include <QTextBlock>

#include "Gui/Settings.h"
#include "Gui/Settings.h"

#include "DataEditorModel.h"
#include "DataEditorController.h"
#include "MetadataEditWidget.h"

MetadataEditWidget::MetadataEditWidget (QWidget *parent) :
    QTextEdit(parent)
{
    QTextEdit::setTextInteractionFlags(Qt::TextSelectableByKeyboard | Qt::TextEditable);
}

void MetadataEditWidget::updateModel ()
{
	auto& cluster = *_model->getClusterToEditRef().metadata;
	auto& cell = *_model->getCellToEditRef().metadata;
	int row = QTextEdit::textCursor().blockNumber();

    //collect new data
    QString currentText = QTextEdit::textCursor().block().text();
    currentText.remove(0, 14);
    if(row == 0)
        cluster.name = currentText;
    if(row == 1)
        cell.name = currentText;

    //inform other instances
	_controller->notificationFromMetadataTab();
}

void MetadataEditWidget::init(DataEditorModel * model, DataEditorController * controller)
{
	_model = model;
	_controller = controller;
}

void MetadataEditWidget::updateDisplay ()
{
	auto const& cluster = *_model->getClusterToEditRef().metadata;
	auto const& cell = *_model->getCellToEditRef().metadata;

	//define auxiliary strings
    QString parStart = "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">";
    QString parEnd = "</p>";
    QString colorTextStart = "<span style=\"color:"+CELL_EDIT_TEXT_COLOR1.name()+"\">";
    QString colorDataStart = "<span style=\"color:"+CELL_EDIT_METADATA_COLOR.name()+"\">";
    QString colorBlack = "<span style=\"color:#000000\">";
    QString colorWhite = "<span style=\"color:#FFFFFF\">";
    QString color1 = "<span style=\"color:"+INDIVIDUAL_CELL_COLOR1.name()+"\">";
    QString color2 = "<span style=\"color:"+INDIVIDUAL_CELL_COLOR2.name()+"\">";
    QString color3 = "<span style=\"color:"+INDIVIDUAL_CELL_COLOR3.name()+"\">";
    QString color4 = "<span style=\"color:"+INDIVIDUAL_CELL_COLOR4.name()+"\">";
    QString color5 = "<span style=\"color:"+INDIVIDUAL_CELL_COLOR5.name()+"\">";
    QString color6 = "<span style=\"color:"+INDIVIDUAL_CELL_COLOR6.name()+"\">";
    QString color7 = "<span style=\"color:"+INDIVIDUAL_CELL_COLOR7.name()+"\">";
    QString colorEnd = "</span>";

//    QString sep0 = colorBlack + "&#9002;" + colorEnd + colorWhite;
//    QString colorBlock1 = "&#8635;";
    QString sep1 = colorBlack + "&#9001;" + colorEnd + color1;
    QString colorBlock2 = "&#9608;&#9608;";
    QString sep2 = colorEnd + colorBlack + "&#9001;" + colorEnd + color2;
    QString colorBlock3 = "&#9608;&#9608;";
    QString sep3 = colorEnd + colorBlack + "&#9001;" + colorEnd + color3;
    QString colorBlock4 = "&#9608;&#9608;";
    QString sep4 = colorEnd + colorBlack + "&#9001;" + colorEnd + color4;
    QString colorBlock5 = "&#9608;&#9608;";
    QString sep5 = colorEnd + colorBlack + "&#9001;" + colorEnd + color5;
    QString colorBlock6 = "&#9608;&#9608;";
    QString sep6 = colorEnd + colorBlack + "&#9001;" + colorEnd + color6;
    QString colorBlock7 = "&#9608;&#9608;";
    QString sep7 = colorEnd + colorBlack + "&#9001;" + colorEnd + color7;
    QString colorBlock8 = "&#9608;&#9608;";//"&#10108;";
    QString sep8 = colorEnd;
    if( cell.color == 0 ) {
        sep1 = colorWhite + "&#9002;" + colorEnd+color1;
        sep2 = colorEnd+ colorWhite + "&#9001;" + colorEnd + color2;
    }
    if(cell.color == 1 ) {
        sep2 = colorEnd + colorWhite + "&#9002;" + colorEnd + color2;
        sep3 = colorEnd + colorWhite + "&#9001;" + colorEnd + color3;
    }
    if(cell.color == 2 ) {
        sep3 = colorEnd + colorWhite + "&#9002;" + colorEnd + color3;
        sep4 = colorEnd + colorWhite + "&#9001;" + colorEnd + color4;
    }
    if(cell.color == 3 ) {
        sep4 = colorEnd + colorWhite + "&#9002;" + colorEnd + color4;
        sep5 = colorEnd + colorWhite + "&#9001;" + colorEnd + color5;
    }
    if(cell.color == 4 ) {
        sep5 = colorEnd + colorWhite + "&#9002;" + colorEnd + color5;
        sep6 = colorEnd + colorWhite + "&#9001;" + colorEnd + color6;
    }
    if(cell.color == 5 ) {
        sep6 = colorEnd + colorWhite + "&#9002;" + colorEnd + color6;
        sep7 = colorEnd + colorWhite + "&#9001;" + colorEnd + color7;
    }
    if(cell.color >= 6 ) {
        sep7 = colorEnd + colorWhite + "&#9002;" + colorEnd + color7;
        sep8 = colorEnd + colorWhite + "&#9001;" + colorEnd;
    }

    //set cursor color
    QPalette p(QTextEdit::palette());
    p.setColor(QPalette::Text, CELL_EDIT_METADATA_CURSOR_COLOR);
    QTextEdit::setPalette(p);

    //create string of display
    QString text;
    text = parStart+colorTextStart+ "cluster name:"+colorEnd;
    text += colorDataStart+" " + cluster.name+colorEnd+parEnd;
    text += parStart+colorTextStart+ "cell name:&nbsp;&nbsp;&nbsp;"+colorEnd;
    text += colorDataStart+" " + cell.name+colorEnd+parEnd;
    text += parStart+colorTextStart+ "cell color: &nbsp;&nbsp;"+colorEnd;
    text += sep1+colorBlock2+sep2+colorBlock3+sep3+colorBlock4+sep4+colorBlock5+sep5+colorBlock6+sep6+colorBlock7+sep7+colorBlock8+sep8+parEnd;

    QTextEdit::setText(text);
}

void MetadataEditWidget::keyPressEvent (QKeyEvent* e)
{
    //notify other instances about update?
    if( (e->key() == Qt::Key_Down) || (e->key() == Qt::Key_Up) || (e->key() == Qt::Key_Enter) || (e->key() == Qt::Key_Return))
        updateModel();

    int col = QTextEdit::textCursor().columnNumber();
    int row = QTextEdit::textCursor().blockNumber();
    int rowLen = QTextEdit::document()->findBlockByNumber(row).length();

    //check for forbidden combinations
    bool forbiddenCombination = false;
    if( e->key() == Qt::Key_PageDown )
        forbiddenCombination = true;
    if( e->key() == Qt::Key_Enter )
        forbiddenCombination = true;
    if( e->key() == Qt::Key_Return )
        forbiddenCombination = true;
    if( col < 15 ) {
        if( e->key() == Qt::Key_Left )
            forbiddenCombination = true;
    }
    if( row == 1 ) {
        if( e->key() == Qt::Key_Down )
            forbiddenCombination = true;
    }
    if( ((col+1) == rowLen) && (e->key() == Qt::Key_Right) )
        forbiddenCombination = true;
    if( rowLen > 37 ) {
        forbiddenCombination = true;
        if( e->key() == Qt::Key_Left )
            forbiddenCombination = false;
        if( e->key() == Qt::Key_Delete )
            forbiddenCombination = false;
        if( e->key() == Qt::Key_Backspace )
            forbiddenCombination = false;
        if( (row == 0) && (e->key() == Qt::Key_Down) )
            forbiddenCombination = false;
        if( (row == 1) && (e->key() == Qt::Key_Up) )
            forbiddenCombination = false;
    }
    if( !forbiddenCombination )
        QTextEdit::keyPressEvent(e);
}

void MetadataEditWidget::mousePressEvent (QMouseEvent* e)
{
    updateModel();
    QTextEdit::mousePressEvent(e);
    int col = QTextEdit::textCursor().columnNumber();
    int row = QTextEdit::textCursor().blockNumber();

    //cluster or cell name clicked?
    if( row < 2 ) {
        if( col < 14 ) {
            QTextEdit::moveCursor(QTextCursor::StartOfBlock);
            QTextEdit::moveCursor(QTextCursor::NextWord);
            QTextEdit::moveCursor(QTextCursor::NextWord);
            QTextEdit::moveCursor(QTextCursor::NextWord);
        }
    }

    //cell color clicked?
    if( row == 2 ) {
		auto& cluster = *_model->getClusterToEditRef().metadata;
		auto& cell = *_model->getCellToEditRef().metadata;
		if ((col == 15) || (col == 16) || (col == 17))
            cell.color = 0;
        if( (col == 18) || (col == 19) || (col == 20) )
            cell.color = 1;
        if( (col == 21) || (col == 22) || (col == 23) )
            cell.color = 2;
        if( (col == 24) || (col == 25) || (col == 26) )
            cell.color = 3;
        if( (col == 27) || (col == 28) || (col == 29) )
            cell.color = 4;
        if( (col == 30) || (col == 31) || (col == 32) )
            cell.color = 5;
        if( (col == 33) || (col == 34) || (col == 35) )
            cell.color = 6;
        updateDisplay();
        QTextEdit::clearFocus();
		_controller->notificationFromMetadataTab();
    }
}

void MetadataEditWidget::mouseDoubleClickEvent (QMouseEvent* e)
{
	QTextEdit::clearFocus();
}




