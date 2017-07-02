#include "CellEdit.h"

#include "gui/Settings.h"
#include "Model/Entities/Cell.h"
#include "Model/Entities/CellCluster.h"

#include <QKeyEvent>
#include <QTextBlock>
#include <QTextLayout>
#include <qmath.h>

CellEdit::CellEdit(QWidget *parent) :
    QTextEdit(parent)
{
    QTextEdit::setTextInteractionFlags(Qt::TextSelectableByKeyboard | Qt::TextEditable);
}

void CellEdit::updateCell (CellTO cell)
{
    _cell = cell;
    updateDisplay();
}

void CellEdit::requestUpdate ()
{
    int row = QTextEdit::textCursor().blockNumber();
    QString currentText = QTextEdit::textCursor().block().text();

    if( row == 0 )
        _cell.cellPos.setX(generateNumberFromFormattedString(currentText));
    if( row == 1 )
        _cell.cellPos.setY(generateNumberFromFormattedString(currentText));
    if( row == 2 )
        _cell.cellEnergy = generateNumberFromFormattedString(currentText);
    if( row == 4 )
        _cell.cellMaxCon = qRound(generateNumberFromFormattedString(currentText));
    if( (row == 6) && _cell.cellAllowToken )
        _cell.cellTokenAccessNum = qRound(generateNumberFromFormattedString(currentText));

    //inform other instances
    Q_EMIT cellDataChanged(_cell);
}

void CellEdit::keyPressEvent (QKeyEvent* e)
{
    //auxilliary data
    QString colorDataStart = "<span style=\"color:"+CELL_EDIT_DATA_COLOR1.name()+"\">";
    QString colorData2Start = "<span style=\"color:"+CELL_EDIT_DATA_COLOR2.name()+"\">";
    QString colorEnd = "</span>";
    int col = QTextEdit::textCursor().columnNumber();
    int row = QTextEdit::textCursor().blockNumber();
    int rowLen = QTextEdit::document()->findBlockByNumber(row).length();

    //request update?
    if( (e->key() == Qt::Key_Down) || (e->key() == Qt::Key_Up) || (e->key() == Qt::Key_Enter) || (e->key() == Qt::Key_Return))
        requestUpdate();

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
        if( rowLen > 37 )
            return;
    }

    //insert number
    if( !k.isEmpty() && (row != 5) ) {
        if( row > 2 )
            QTextEdit::textCursor().insertHtml(colorDataStart+k+colorEnd);
        else {
            int s = QTextEdit::textCursor().block().text().indexOf(".");

            //no dot before cursor?
            if( (s < 0) || (s >= col) )
                QTextEdit::textCursor().insertHtml(colorDataStart+k+colorEnd);
            else
                QTextEdit::textCursor().insertHtml(colorData2Start+k+colorEnd);
        }
    }

    //typing dot?
    if( e->key() == Qt::Key_Period ) {
        if( row < 3 ) {
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
                QString t = QTextEdit::document()->findBlockByLineNumber(row).text().right(n).remove('.');//.left(s-col);
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
        if( row != 5) {

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
    }

    //typing y or n?
    if( row == 5 ){
        k = "";
        if( e->key() == Qt::Key_Y ) {
            k = "y";
            _cell.cellAllowToken = true;
        }
        if( e->key() == Qt::Key_N ) {
            k = "n";
            _cell.cellAllowToken = false;
        }
        if( !k.isEmpty() ) {
            QTextEdit::textCursor().deleteChar();
            QTextEdit::textCursor().insertHtml(colorDataStart+k+colorEnd);
            QTextEdit::moveCursor(QTextCursor::Left);
            updateDisplay();

            //inform other instances
            Q_EMIT cellDataChanged(_cell);
        }
    }

    //typing left button?
    if( (col > 24) && (e->key() == Qt::Key_Left) )
        if( row != 5)
            QTextEdit::keyPressEvent(e);

    //typing right button?
    if( (col < rowLen-1) && (e->key() == Qt::Key_Right) )
        if( row != 5)
            QTextEdit::keyPressEvent(e);

    //typing down button?
    if( e->key() == Qt::Key_Down ) {
        if( (row <= 1) || ((row >= 3) && (row <= 4)) )
            QTextEdit::keyPressEvent(e);
        if( row == 2) {
            QTextEdit::moveCursor(QTextCursor::Down);
            QTextEdit::keyPressEvent(e);
        }
        if( (row == 5) && _cell.cellAllowToken )
            QTextEdit::keyPressEvent(e);
        if( row == 4 ) {
            QTextEdit::moveCursor(QTextCursor::EndOfLine);
            QTextEdit::moveCursor(QTextCursor::PreviousWord);
        }
    }

    //typing up button?
    if( e->key() == Qt::Key_Up ) {
        if( row == 0 ) {
            QTextEdit::moveCursor(QTextCursor::Up);
            QTextEdit::moveCursor(QTextCursor::Up);
            QTextEdit::keyPressEvent(e);
        }
        if( ((row >= 1) && (row <= 3)) || ((row >= 5) && (row <= 6)) )
            QTextEdit::keyPressEvent(e);
        if( row == 4 ) {
            QTextEdit::moveCursor(QTextCursor::Up);
            QTextEdit::keyPressEvent(e);
        }
        if( row == 6 ) {
            QTextEdit::moveCursor(QTextCursor::EndOfLine);
            QTextEdit::moveCursor(QTextCursor::PreviousWord);
        }
    }
}

void CellEdit::mousePressEvent(QMouseEvent* e)
{
    QTextEdit::mousePressEvent(e);
    int col = QTextEdit::textCursor().columnNumber();
    int row = QTextEdit::textCursor().blockNumber();

    //move cursor to correct position
    if( row <= 2 ) {
        if( col < 24 ) {
            QTextEdit::moveCursor(QTextCursor::StartOfBlock);
            QTextEdit::moveCursor(QTextCursor::NextWord);
            QTextEdit::moveCursor(QTextCursor::NextWord);
            QTextEdit::moveCursor(QTextCursor::NextWord);
        }
    }
    if( row == 3 ) {
        QTextEdit::clearFocus();
    }
    if( (row >= 4) && (row <= 5) ) {
        QTextEdit::moveCursor(QTextCursor::StartOfBlock);
        QTextEdit::moveCursor(QTextCursor::NextWord);
        QTextEdit::moveCursor(QTextCursor::NextWord);
        QTextEdit::moveCursor(QTextCursor::NextWord);
    }
    if( row == 6 ) {
        if( _cell.cellAllowToken ) {
            QTextEdit::moveCursor(QTextCursor::StartOfBlock);
            QTextEdit::moveCursor(QTextCursor::NextWord);
            QTextEdit::moveCursor(QTextCursor::NextWord);
            QTextEdit::moveCursor(QTextCursor::NextWord);
        }
        else
            QTextEdit::clearFocus();
    }

    if( row >= 7 ) {
        QTextEdit::clearFocus();
    }

    //cursor at cell function?
    if( (row >= 7) && (col >= 18) && (col <= 36)) {
        if( row == 7 )
            _cell.cellFunctionType = Enums::CellFunction::COMPUTER;
        if( row == 8 )
            _cell.cellFunctionType = Enums::CellFunction::PROPULSION;
        if( row == 9 )
            _cell.cellFunctionType = Enums::CellFunction::SCANNER;
        if( row == 10 )
            _cell.cellFunctionType = Enums::CellFunction::WEAPON;
        if( row == 11 )
            _cell.cellFunctionType = Enums::CellFunction::CONSTRUCTOR;
        if( row == 12 )
            _cell.cellFunctionType = Enums::CellFunction::SENSOR;
        if( row == 13 )
            _cell.cellFunctionType = Enums::CellFunction::COMMUNICATOR;
        updateDisplay();

        //inform other instances
        Q_EMIT cellDataChanged(_cell);
    }
}

void CellEdit::mouseDoubleClickEvent (QMouseEvent* e)
{
    QTextEdit::clearFocus();
}

void CellEdit::wheelEvent (QWheelEvent* e)
{
    QTextEdit::wheelEvent(e);
    QTextEdit::clearFocus();
}

void CellEdit::updateDisplay ()
{
    int col = QTextEdit::textCursor().columnNumber();
    int row = QTextEdit::textCursor().blockNumber();

    //define auxilliary strings
    QString parStart = "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">";
    QString parEnd = "</p>";
    QString colorCaptionStart = "<span style=\"color:"+CELL_EDIT_CAPTION_COLOR1.name()+"\">";
    QString colorTextStart = "<span style=\"color:"+CELL_EDIT_TEXT_COLOR1.name()+"\">";
    QString colorTextStartInactive = "<span style=\"color:"+CELL_EDIT_TEXT_COLOR2.name()+"\">";
    QString colorDataStart = "<span style=\"color:"+CELL_EDIT_DATA_COLOR1.name()+"\">";
    QString colorEnd = "</span>";
    QString text;

    //set cursor color
    QPalette p(QTextEdit::palette());
    p.setColor(QPalette::Text, CELL_EDIT_CURSOR_COLOR);
    QTextEdit::setPalette(p);

    //create string of display
    text = parStart+colorTextStart+ "position x: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"+colorEnd;
    text += generateFormattedRealString(_cell.cellPos.x())+parEnd;
    text += parStart+colorTextStart+ "position y: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"+colorEnd;
    text += generateFormattedRealString(_cell.cellPos.y())+parEnd;
    text += parStart+colorTextStart+ "internal energy: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"+colorEnd;
    text += generateFormattedRealString(_cell.cellEnergy)+parEnd;
    text += parStart+colorTextStart+ "current bonds: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"+colorEnd;
    text += colorDataStart+QString("%1").arg(_cell.cellNumCon)+colorEnd+parEnd;
    text += parStart+colorTextStart+ "max bonds: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"+colorEnd;
    text += colorDataStart+QString("%1").arg(_cell.cellMaxCon)+colorEnd+parEnd;
    text += parStart+colorTextStart+ "allow token: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"+colorEnd;
    if( _cell.cellAllowToken ) {
        text += colorDataStart+"y"+colorEnd+parEnd;
        text += parStart+colorTextStart+ "branch number: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"+colorEnd;
        text += colorDataStart+QString("%1").arg(_cell.cellTokenAccessNum)+colorEnd+parEnd;
    }
    else {
        text += colorDataStart+"n"+colorEnd+parEnd;
        text += parStart+colorTextStartInactive+ "branch number: &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"+colorEnd+parEnd;
    }
    text += generateFormattedCellFunctionString(_cell.cellFunctionType);

    QTextEdit::setText(text);

    //restore cursor
    for( int i = 0; i < row; ++i )
        QTextEdit::moveCursor(QTextCursor::NextBlock);
    for( int i = 0; i < col; ++i )
        QTextEdit::moveCursor(QTextCursor::Right);
}

qreal CellEdit::generateNumberFromFormattedString (QString s)
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

QString CellEdit::generateFormattedRealString (QString s)
{
    QString colorDataStart = "<span style=\"color:"+CELL_EDIT_DATA_COLOR1.name()+"\">";
    QString colorData2Start = "<span style=\"color:"+CELL_EDIT_DATA_COLOR2.name()+"\">";
    QString colorEnd = "</span>";
    QString iS, reS;
    int i = s.indexOf(".");
    if( i == -1 )
        iS = s;
    else {
        iS = s.left(i);
        reS = s.remove(0,i+1);
    }
    return colorDataStart+iS+colorEnd+colorData2Start+"."+reS+colorEnd;
}

QString CellEdit::generateFormattedRealString (qreal r)
{
    QString colorDataStart = "<span style=\"color:"+CELL_EDIT_DATA_COLOR1.name()+"\">";
    QString colorData2Start = "<span style=\"color:"+CELL_EDIT_DATA_COLOR2.name()+"\">";
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

QString CellEdit::generateFormattedCellFunctionString (Enums::CellFunction::Type type)
{
    //define auxilliary strings
    QString parStart = "<p style=\" margin-top:0px; margin-bottom:0px; margin-left:0px; margin-right:0px; -qt-block-indent:0; text-indent:0px;\">";
    QString parEnd = "</p>";
    QString colorTextStart = "<span style=\"color:"+CELL_EDIT_TEXT_COLOR1.name()+"\">";
    QString colorDataStart = "<span style=\"color:"+CELL_EDIT_DATA_COLOR1.name()+"\">";
    QString colorData2Start = "<span style=\"color:"+CELL_EDIT_DATA_COLOR2.name()+"\">";
    QString colorEnd = "</span>";
    QString text;

    //generate formatted string
    text += parStart+colorTextStart+ "cell function:&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;"+colorEnd;
    if( type == Enums::CellFunction::COMPUTER )
        text += colorDataStart+"&nbsp;&#9002; computer &#9001;&nbsp;"+colorEnd+parEnd;
    else
        text += colorData2Start+"&nbsp;&nbsp;&nbsp;computer"+colorEnd+parEnd;
    if( type == Enums::CellFunction::PROPULSION )
        text += parStart+colorDataStart+"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#9002; propulsion &#9001;&nbsp;"+colorEnd+parEnd;
    else
        text += parStart+colorData2Start+"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;propulsion"+colorEnd+parEnd;
    if( type == Enums::CellFunction::SCANNER )
        text += parStart+colorDataStart+"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#9002; scanner &#9001;&nbsp;"+colorEnd+parEnd;
    else
        text += parStart+colorData2Start+"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;scanner"+colorEnd+parEnd;
    if( type == Enums::CellFunction::WEAPON )
        text += parStart+colorDataStart+"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#9002; weapon &#9001;&nbsp;"+colorEnd+parEnd;
    else
        text += parStart+colorData2Start+"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;weapon"+colorEnd+parEnd;
    if( type == Enums::CellFunction::CONSTRUCTOR )
        text += parStart+colorDataStart+"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#9002; constructor &#9001;&nbsp;"+colorEnd+parEnd;
    else
        text += parStart+colorData2Start+"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;constructor"+colorEnd+parEnd;
    if( type == Enums::CellFunction::SENSOR )
        text += parStart+colorDataStart+"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#9002; sensor &#9001;&nbsp;"+colorEnd+parEnd;
    else
        text += parStart+colorData2Start+"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;sensor"+colorEnd+parEnd;
    if( type == Enums::CellFunction::COMMUNICATOR )
        text += parStart+colorDataStart+"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&#9002; communicator &#9001;&nbsp;"+colorEnd+parEnd;
    else
        text += parStart+colorData2Start+"&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;communicator"+colorEnd+parEnd;
    return text;
}

