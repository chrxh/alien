#include <QKeyEvent>
#include <QTextBlock>
#include <qmath.h>

#include "Gui/Settings.h"

#include "DataEditModel.h"
#include "DataEditController.h"
#include "TokenEditWidget.h"


TokenEditWidget::TokenEditWidget(QWidget *parent)
    : QTextEdit(parent)
{
    QTextEdit::setTextInteractionFlags(Qt::TextSelectableByKeyboard | Qt::TextEditable);
}

void TokenEditWidget::init(DataEditModel * model, DataEditController * controller, int tokenIndex)
{
	_model = model;
	_controller = controller;
	_tokenIndex = tokenIndex;
}

void TokenEditWidget::updateDisplay ()
{
	auto const& token = _model->getTokenToEditRef(_tokenIndex);

    //define auxilliary strings
    QString parStart = "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">";
    QString parEnd = "</p>";
    QString colorCaptionStart = "<span style=\"color:"+Const::CellEditCaptionColor1.name()+"\">";
    QString colorTextStart = "<span style=\"color:"+Const::CellEditTextColor1.name()+"\">";
    QString colorEnd = "</span>";
    QString text;

    //set cursor color
    QPalette p(QTextEdit::palette());
    p.setColor(QPalette::Text, Const::CellEditCursorColor);
    QTextEdit::setPalette(p);

    //create string of display
    text = parStart+colorTextStart+ "internal energy: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"+colorEnd;
    text += generateFormattedRealString(*token.energy)+parEnd;
    QTextEdit::setText(text);
}

void TokenEditWidget::requestUpdate ()
{
	auto& token = _model->getTokenToEditRef(_tokenIndex);
	QString currentText = QTextEdit::textCursor().block().text();
    token.energy = generateNumberFromFormattedString(currentText);

	_controller->notificationFromTokenTab();
}

void TokenEditWidget::keyPressEvent (QKeyEvent* e)
{
    //auxilliary data
    QString colorDataStart = "<span style=\"color:"+Const::CellEditDataColor1.name()+"\">";
    QString colorData2Start = "<span style=\"color:"+Const::CellEditDataColor2.name()+"\">";
    QString colorEnd = "</span>";
    int col = QTextEdit::textCursor().columnNumber();
    int row = QTextEdit::textCursor().blockNumber();
    int rowLen = QTextEdit::document()->findBlockByNumber(row).length();

    //typing number?
    QString k;
    if( (e->key() == Qt::Key_0) )
        k = "0";
    if( (e->key() == Qt::Key_1) )
        k = "1";
    if( (e->key() == Qt::Key_2) )
        k = "2";
    if( (e->key() == Qt::Key_3) )
        k = "3";
    if( (e->key() == Qt::Key_4) )
        k = "4";
    if( (e->key() == Qt::Key_5) )
        k = "5";
    if( (e->key() == Qt::Key_6) )
        k = "6";
    if( (e->key() == Qt::Key_7) )
        k = "7";
    if( (e->key() == Qt::Key_8) )
        k = "8";
    if( (e->key() == Qt::Key_9) )
        k = "9";

    //check for end of line in the case typing number or period
    if( (!k.isEmpty()) || (e->key() == Qt::Key_Period) ) {
        if( QTextEdit::textCursor().block().text().length() > 36 )
            return;
    }

    //typing left button?
    if( (col > 24) && (e->key() == Qt::Key_Left) )
        QTextEdit::keyPressEvent(e);

    //typing right button?
    if( (col < rowLen-1) && (e->key() == Qt::Key_Right) )
        QTextEdit::keyPressEvent(e);

    //insert number
    if( !k.isEmpty() ) {
        int s = QTextEdit::textCursor().block().text().indexOf(".");

        //no dot before cursor?
        if( (s < 0) || (s >= col) )
            QTextEdit::textCursor().insertHtml(colorDataStart+k+colorEnd);
        else
            QTextEdit::textCursor().insertHtml(colorData2Start+k+colorEnd);
    }

    //typing dot?
    if( e->key() == Qt::Key_Period ) {
        int s = QTextEdit::textCursor().block().text().indexOf(".");

        //there is a dot after cursor?
        if( (s < 0) || (s >= col) ) {
            QTextEdit::textCursor().insertHtml(colorData2Start+"."+colorEnd);

            //removes other dots and recolor the characters from new dot to line end
            int n = rowLen-col-1;   //number of characters from dot to line end
            QString t = QTextEdit::document()->findBlockByLineNumber(row).text().right(n).remove('.');//.left(s-col);
            for( int i = 0; i < n; ++i )
                QTextEdit::textCursor().deleteChar();
            QTextEdit::textCursor().insertHtml(colorData2Start+t+colorEnd);
            for( int i = 0; i < t.size(); ++i )
                QTextEdit::moveCursor(QTextCursor::Left);
        }

        //there is a dot before cursor?
        if( (s >= 0) && (s < col) ) {

            //removes other dots and recolor the characters from old dot to line end
            int n = rowLen-s-1;   //number of characters from dot to line end
            QString t = QTextEdit::document()->findBlockByLineNumber(row).text().right(n).remove('.');
            QTextCursor c = QTextEdit::textCursor();
            c.movePosition(QTextCursor::Left,QTextCursor::MoveAnchor, col-s);
            for( int i = 0; i < n; ++i )
                c.deleteChar();
            c.insertHtml(colorDataStart+t.left(col-s-1)+colorEnd);
            c.insertHtml(colorData2Start+"."+t.right(n-col+s)+colorEnd);
            for( int i = 0; i < n-col+s; ++i )
                QTextEdit::moveCursor(QTextCursor::Left);
        }
    }

    //typing backspace?
    if( (col > 24) && (e->key() == Qt::Key_Backspace) ) {

        //is there a dot at cursor's position?
        QChar c = QTextEdit::document()->characterAt(QTextEdit::textCursor().position()-1);
        if( c == '.' ) {
            QTextEdit::keyPressEvent(e);

            //recolor the characters from dot to line end
            int n = rowLen-col-1;   //number of characters from dot to line end
            QString t = QTextEdit::document()->findBlockByLineNumber(row).text().right(n);
            for( int i = 0; i < n; ++i )
                QTextEdit::textCursor().deleteChar();
            QTextEdit::textCursor().insertHtml(colorDataStart+t+colorEnd);
            for( int i = 0; i < n; ++i )
                QTextEdit::moveCursor(QTextCursor::Left);
        }
        else
            QTextEdit::keyPressEvent(e);
    }

    //typing delete?
    if( (col < rowLen-1) && (e->key() == Qt::Key_Delete) ) {

        //is there a dot at cursor's position?
        QChar c = QTextEdit::document()->characterAt(QTextEdit::textCursor().position());
        if( c == '.' ) {
            QTextEdit::keyPressEvent(e);

            //recolor the characters from dot to line end
            int n = rowLen-col-2;   //number of characters from dot to line end
            QString t = QTextEdit::document()->findBlockByLineNumber(row).text().right(n);
            for( int i = 0; i < n; ++i )
                QTextEdit::textCursor().deleteChar();
            QTextEdit::textCursor().insertHtml(colorDataStart+t+colorEnd);
            for( int i = 0; i < n; ++i )
                QTextEdit::moveCursor(QTextCursor::Left);
        }
        else
            QTextEdit::keyPressEvent(e);
    }

    if( (e->key() == Qt::Key_Down) || (e->key() == Qt::Key_Up) || (e->key() == Qt::Key_Enter) || (e->key() == Qt::Key_Return) ) {

        //inform other instances
        QString energyStr = QTextEdit::textCursor().block().text();
		auto& token = _model->getTokenToEditRef(_tokenIndex);
		token.energy = generateNumberFromFormattedString(energyStr);
		_controller->notificationFromTokenTab();

    }
}

void TokenEditWidget::mousePressEvent(QMouseEvent* e)
{
    QTextEdit::mousePressEvent(e);

    //move cursor to correct position
    int col = QTextEdit::textCursor().columnNumber();
    int row = QTextEdit::textCursor().blockNumber();
    if( (row == 0) && (col < 24) ) {
        QTextEdit::moveCursor(QTextCursor::StartOfBlock);
        QTextEdit::moveCursor(QTextCursor::NextWord);
        QTextEdit::moveCursor(QTextCursor::NextWord);
        QTextEdit::moveCursor(QTextCursor::NextWord);
    }
}

void TokenEditWidget::mouseDoubleClickEvent (QMouseEvent* e)
{
    QTextEdit::clearFocus();
}

void TokenEditWidget::wheelEvent (QWheelEvent* e)
{
    QTextEdit::wheelEvent(e);
    QTextEdit::clearFocus();
}

qreal TokenEditWidget::generateNumberFromFormattedString (QString s)
{
    int i = s.indexOf(':');
    if( i >= 0 ) {
        QString sub = s.right(s.size()-i-1);
        qreal d = sub.toDouble();
        if( d >= 0.0 )
            d += 0.00005;
        else
            d -= 0.00005;
        return d;
    }
    return 0.0;
}

QString TokenEditWidget::generateFormattedRealString (qreal r)
{
    //define auxilliary strings
    QString colorDataStart = "<span style=\"color:"+Const::CellEditDataColor1.name()+"\">";
    QString colorData2Start = "<span style=\"color:"+Const::CellEditDataColor2.name()+"\">";
    QString colorEnd = "</span>";

    bool negativeSign = false;
    if( r < 0.0 ) {
        r = -r;
        negativeSign = true;
    }
    int i = qFloor(r);
    int re = (r-qFloor(r))*10000.0;
    QString iS = QString("%1").arg(i);
    QString reS = QString("%1").arg(re, 4);
    reS.replace(" ", "0");
    if( negativeSign)
        return colorDataStart+"-"+iS+colorEnd+colorData2Start+"."+reS+colorEnd;
    else
        return colorDataStart+iS+colorEnd+colorData2Start+"."+reS+colorEnd;
}




