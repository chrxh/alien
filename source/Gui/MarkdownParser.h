#pragma once

#include <string>

#include "MarkdownDocument.h"

class MarkdownParser
{
public:
    static MarkdownDocument parse(std::string const& text);

    static std::string toPlainText(MarkdownSpans const& spans);

    // GitHub-style anchor, e.g. "Cell functions" -> "cell-functions"
    static std::string toAnchor(std::string const& headingText);
};
