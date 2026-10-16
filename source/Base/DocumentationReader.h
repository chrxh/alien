#pragma once

#include <string>
#include <vector>

struct DocumentationChapter
{
    bool operator==(DocumentationChapter const&) const = default;

    std::string filename;
    std::string title;
    std::string part;
};

class DocumentationReader
{
public:
    static std::vector<DocumentationChapter> getChapters();
    static std::vector<DocumentationChapter> parseChapters(std::string const& contentsText);
    static std::string readChapter(DocumentationChapter const& chapter);
};
