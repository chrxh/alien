#pragma once

#include "Definitions.h"
#include "Settings.h"

class StringHelper
{
public:
	static QString toString(int value)
	{
		return QString::fromStdString(std::to_string(value));
	}

	static QString toString(double value)
	{
		return QString("%1").arg(value);// QString::fromStdString(std::to_string(value));
	}

	static QString generateFormattedIntString(int integer)
	{
		QString colorDataStart = "<span style=\"color:" + Const::CellEditDataColor1.name() + "\">";
		QString colorEnd = "</span>";
        QString integerAsString;

        do {
            if (!integerAsString.isEmpty()) {
                integerAsString = QString(",") + integerAsString;
            }
            if (integer >= 1000) {
                integerAsString = QString("%1").arg(integer % 1000, 3) + integerAsString;
            }
            else {
                integerAsString = QString("%1").arg(integer % 1000) + integerAsString;
            }
            integer = integer / 1000;
        } while (integer > 0);
        integerAsString.replace(" ", "0");

        return colorDataStart + integerAsString + colorEnd;
	}

	static QString generateFormattedRealString(qreal r)
	{
		QString colorDataStart = "<span style=\"color:" + Const::CellEditDataColor1.name() + "\">";
		QString colorData2Start = "<span style=\"color:" + Const::CellEditDataColor2.name() + "\">";
		QString colorEnd = "</span>";
		bool negativeSign = false;
		if (r < 0.0) {
			r = -r;
			negativeSign = true;
		}
		int i = qFloor(r);
		int re = (r - qFloor(r))*10000.0;
		QString iS = generateFormattedIntString(i);
		QString reS = QString("%1").arg(re, 4);
		reS.replace(" ", "0");
		if (negativeSign)
			return colorDataStart + "-" + iS + colorEnd + colorData2Start + "." + reS + colorEnd;
		else
			return colorDataStart + iS + colorEnd + colorData2Start + "." + reS + colorEnd;
	}

	static QString ws(int num)
	{
		QString result;
		for (int i = 0; i < num; ++i) {
			result += "&nbsp;";
		}
		return result;
	}
};
