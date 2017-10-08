#include "HexEditWidget.h"

#include"gui/Settings.h"

#include <QTextDocument>
#include <QTextBlock>
#include <QKeyEvent>

HexEditWidget::HexEditWidget(QWidget *parent) :
    QTextEdit(parent)
{
    QTextEdit::setTextInteractionFlags(Qt::TextSelectableByKeyboard | Qt::TextEditable);
    QTextEdit::setCursorWidth(6);
}

HexEditWidget::~HexEditWidget ()
{

}

void HexEditWidget::update ()
{
    QTextEdit::setText("");
}

void HexEditWidget::update (QByteArray const& data)
{
    _data = data;
    displayData();
}

QByteArray const& HexEditWidget::getDataRef ()
{
    return _data;
}

QByteArray HexEditWidget::convertHexStringToByteArray(QString hexCode)
{
    QByteArray d(hexCode.size()/2, 0);
    int len = hexCode.length()/2;
    for(int i = 0; i < len; ++i ) {
        bool ok = true;
        d[i] = hexCode.left(2).toUInt(&ok, 16);
        hexCode.remove(0, 2);
    }
    return d;
}

void HexEditWidget::keyPressEvent (QKeyEvent* e)
{
//    QTextEdit::keyPressEvent(e);

    //read valid keys
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
    if( (e->key() == Qt::Key_A) )
        k = "A";
    if( (e->key() == Qt::Key_B) )
        k = "B";
    if( (e->key() == Qt::Key_C) )
        k = "C";
    if( (e->key() == Qt::Key_D) )
        k = "D";
    if( (e->key() == Qt::Key_E) )
        k = "E";
    if( (e->key() == Qt::Key_F) )
        k = "F";

    //valid key?
    if( !k.isEmpty() && (!QTextEdit::textCursor().atEnd()) ) {

        //set char
        QTextEdit::textCursor().deleteChar();
        int pos = QTextEdit::textCursor().positionInBlock();
        QString s1;
        if( ((pos % 6) == 0) || ((pos % 6) == 1))
            s1 = "<span style=\"color:"+HEX_EDIT_COLOR1.name()+"\">";
        else if( ((pos % 6) == 3) || ((pos % 6) == 4))
            s1 = "<span style=\"color:"+HEX_EDIT_COLOR2.name()+"\">";
        QTextEdit::textCursor().insertHtml(s1+k+"</span>");

        //read and Q_EMIT data
        QString data = QTextEdit::document()->findBlockByLineNumber(0).text();
        data.remove(" ");
        _data = convertHexStringToByteArray(data);
        Q_EMIT dataChanged(_data);
    }

    //arrow keys pressed?
    if( e->key() == Qt::Key_Right ) {
        QTextEdit::moveCursor(QTextCursor::NextCharacter);
    }
    if( e->key() == Qt::Key_Left ) {
        if( QTextEdit::textCursor().positionInBlock() == 0 )
            Q_EMIT cursorReachedBeginning(-1);
        else {
            QTextEdit::moveCursor(QTextCursor::Left);
            QTextEdit::moveCursor(QTextCursor::Left);
        }

/*        if( QTextEdit::textCursor().blockNumber() == 0 )
            QTextEdit::moveCursor(QTextCursor::Right);
        if( QTextEdit::textCursor().blockNumber() == 0 )
            QTextEdit::moveCursor(QTextCursor::Right);*/
    }
    if( e->key() == Qt::Key_Up ) {
        int oldPos = QTextEdit::textCursor().positionInBlock();
        QTextEdit::moveCursor(QTextCursor::Up);
        int pos = QTextEdit::textCursor().positionInBlock();
        if( oldPos == pos )
            Q_EMIT cursorReachedBeginning(QTextEdit::textCursor().columnNumber());
//        if( QTextEdit::textCursor().blockNumber() == 0 )
//            QTextEdit::moveCursor(QTextCursor::Down);
    }
    if( e->key() == Qt::Key_Down ) {
        int oldPos = QTextEdit::textCursor().positionInBlock();
        QTextEdit::moveCursor(QTextCursor::Down);
        int pos = QTextEdit::textCursor().positionInBlock();
        if( oldPos == pos )
            Q_EMIT cursorReachedEnd(QTextEdit::textCursor().columnNumber());
    }

    //skip the empty space
    int pos = QTextEdit::textCursor().positionInBlock();
    if( (pos % 3) == 2 )
        QTextEdit::moveCursor(QTextCursor::NextWord);
    if( pos == (_data.size()*3-1) ) {
        QTextEdit::moveCursor(QTextCursor::Left);
        QTextEdit::moveCursor(QTextCursor::Left);
        Q_EMIT cursorReachedEnd(0);
    }

    //adapt color of the cursor
    pos = QTextEdit::textCursor().positionInBlock();
    QPalette p(QTextEdit::palette());
    if( ((pos % 6) == 0) || ((pos % 6) == 1) )
        p.setColor(QPalette::Text, HEX_EDIT_COLOR1);
    else if( ((pos % 6) == 3) || ((pos % 6) == 4) )
        p.setColor(QPalette::Text, HEX_EDIT_COLOR2);
    QTextEdit::setPalette(p);

}

void HexEditWidget::mousePressEvent(QMouseEvent* e)
{
    QTextEdit::mousePressEvent(e);
//    if( QTextEdit::textCursor().blockNumber() == 0 )
//        QTextEdit::moveCursor(QTextCursor::Down);

    //skip the empty space
    int pos(QTextEdit::textCursor().positionInBlock());
    if( (pos % 3) == 2 )
        QTextEdit::moveCursor(QTextCursor::NextWord);
    if( pos > (_data.size()*3-2) ) {
        QTextEdit::moveCursor(QTextCursor::Left);
        QTextEdit::moveCursor(QTextCursor::Left);
    }

    //adapt color of the cursor
    pos = QTextEdit::textCursor().positionInBlock();
    QPalette p(QTextEdit::palette());
    if( ((pos % 6) == 0) || ((pos % 6) == 1) )
        p.setColor(QPalette::Text, HEX_EDIT_COLOR1);
    else if( ((pos % 6) == 3) || ((pos % 6) == 4) )
        p.setColor(QPalette::Text, HEX_EDIT_COLOR2);
    QTextEdit::setPalette(p);
}

void HexEditWidget::mouseDoubleClickEvent (QMouseEvent* e)
{

}

void HexEditWidget::wheelEvent (QWheelEvent* e)
{
    QTextEdit::wheelEvent(e);
    QTextEdit::clearFocus();
}

void HexEditWidget::displayData ()
{
    int col = QTextEdit::textCursor().columnNumber();
    int row = QTextEdit::textCursor().blockNumber();

    //define auxilliary strings
    QString parStart = "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">";
    QString parEnd = "</p>";
    QString colorEnd = "</span>";
    QString text;

    //create string of display
//    text = colorCaptionStart+parStart+_caption+parEnd+colorEnd;

    text = parStart;
    for(int i = 0; i < _data.size(); ++i ) {
		quint8 byte = _data[i];
        QString s1(QString("%1").arg(byte >>4, 1, 16, QLatin1Char('0')));
        QString s2(QString("%1").arg(byte &15, 1, 16, QLatin1Char('0')));
        if( (i%2) == 0 )
            text += "<span style=\"color:"+HEX_EDIT_COLOR1.name()+"\">"+s1.toUpper()+s2.toUpper()+"  "+colorEnd;
        else
            text += "<span style=\"color:"+HEX_EDIT_COLOR2.name()+"\">"+s1.toUpper()+s2.toUpper()+" "+colorEnd;
    }
    text += parEnd;
    QTextEdit::setText(text);

    //restore cursor
    for( int i = 0; i < row; ++i )
        QTextEdit::moveCursor(QTextCursor::NextBlock);
    for( int i = 0; i < col; ++i )
        QTextEdit::moveCursor(QTextCursor::Right);
}


