#include "NetworkResourceService.h"

#include <ranges>

#include <boost/algorithm/string/split.hpp>
#include <boost/algorithm/string.hpp>
#include <boost/range/adaptor/indexed.hpp>

#include "NetworkResourceRawTO.h"
#include "NetworkResourceTreeTO.h"

namespace
{
    int getNumEqualFolders(std::vector<std::string> const& folderNames, std::vector<std::string> const& otherFolderNames)
    {
        auto equalFolders = 0;
        auto numFolders = std::min(folderNames.size(), otherFolderNames.size());
        for (int i = 0; i < numFolders; ++i) {
            if (folderNames[i] == otherFolderNames[i]) {
                ++equalFolders;
            } else {
                return equalFolders;
            }
        }
        return equalFolders;
    }

    //returns true iff folderNames contains otherFolderNames
    bool contains(std::vector<std::string> const& folderNames, std::vector<std::string> const& otherFolderNames)
    {
        if (folderNames.size() < otherFolderNames.size()) {
            return false;
        }
        for (size_t i = 0; i < otherFolderNames.size(); ++i) {
            if (folderNames.at(i) != otherFolderNames.at(i)) {
                return false;
            }
        }
        return true;
    }
}

std::unordered_map<NetworkResourceTreeTO, std::vector<NetworkResourceRawTO>> NetworkResourceService::_treeTOtoRawTOcache;

std::vector<NetworkResourceTreeTO> NetworkResourceService::createTreeTOs(
    std::vector<NetworkResourceRawTO> const& rawTOs,
    std::set<std::vector<std::string>> const& collapsedFolderNames)
{
    NetworkResourceService::invalidateCache();

    std::list<NetworkResourceTreeTO> treeToList;
    for (auto const& rawTO : rawTOs) {

        //parse folder names
        std::vector<std::string> folderNames;
        std::string nameWithoutFolders;
        boost::split(folderNames, rawTO->resourceName, boost::is_any_of("/"));
        if (!folderNames.empty()) {
            nameWithoutFolders = folderNames.back();
            folderNames.pop_back();
        }

        std::list<NetworkResourceTreeTO>::iterator bestMatchIter;
        int bestMatchEqualFolders;
        if (!treeToList.empty()) {

            //find matching node
            auto searchIter = treeToList.end();
            bestMatchIter = searchIter;
            bestMatchEqualFolders = -1;
            for (int i = 0; i < treeToList.size(); ++i) {
                --searchIter;
                auto otherRawTO = *searchIter;
                auto equalFolders = getNumEqualFolders(folderNames, otherRawTO->folderNames);
                if (equalFolders < bestMatchEqualFolders) {
                    break;
                }
                if (equalFolders > bestMatchEqualFolders) {
                    bestMatchIter = searchIter;
                    bestMatchEqualFolders = equalFolders;
                }
            }
            ++bestMatchIter;
        } else {
            bestMatchIter = treeToList.begin();
            bestMatchEqualFolders = 0;
        }

        //insert folders
        for (int i = bestMatchEqualFolders; i < folderNames.size(); ++i) {
            auto treeTO = std::make_shared<_NetworkResourceTreeTO>();
            treeTO->folderNames = std::vector(folderNames.begin(), folderNames.begin() + i + 1);
            treeTO->type = rawTO->resourceType;
            treeTO->node = BrowserFolder();
            bestMatchIter = treeToList.insert(bestMatchIter, treeTO);
            ++bestMatchIter;
        }

        //insert leaf
        auto treeTO = std::make_shared<_NetworkResourceTreeTO>();
        BrowserLeaf leaf{.leafName = nameWithoutFolders, .rawTO = rawTO};
        treeTO->type = rawTO->resourceType;
        treeTO->folderNames = folderNames;
        treeTO->node = leaf;
        treeToList.insert(bestMatchIter, treeTO);
    }

    //calc folder lines
    std::vector treeTOs(treeToList.begin(), treeToList.end());
    for (int i = 0; i < treeTOs.size(); ++i) {
        auto& treeTO = treeTOs.at(i);

        if (i == 0) {
            if (!treeTO->isLeaf()) {
                treeTO->treeSymbols.emplace_back(FolderTreeSymbols::Expanded);
            }
        } else {
            auto const& prevTreeTO = treeTOs.at(i - 1);
            auto numEqualFolders = getNumEqualFolders(treeTO->folderNames, prevTreeTO->folderNames);

            treeTO->treeSymbols.resize(treeTO->folderNames.size(), FolderTreeSymbols::None);

            //calc symbols at position numEqualFolders - 1
            if (numEqualFolders > 0) {
                int f = numEqualFolders - 1;
                if (prevTreeTO->treeSymbols.at(f) == FolderTreeSymbols::Expanded) {
                    treeTO->treeSymbols.at(f) = FolderTreeSymbols::End;
                } else if (prevTreeTO->treeSymbols.at(f) == FolderTreeSymbols::End) {
                    prevTreeTO->treeSymbols.at(f) = FolderTreeSymbols::Branch;
                    treeTO->treeSymbols.at(f) = FolderTreeSymbols::End;
                } else if (prevTreeTO->treeSymbols.at(f) == FolderTreeSymbols::Branch) {
                    treeTO->treeSymbols.at(f) = FolderTreeSymbols::End;
                } else if (prevTreeTO->treeSymbols.at(f) == FolderTreeSymbols::None) {
                    for (int j = i - 1; j >= 0; --j) {
                        auto& otherTreeTO = treeTOs.at(j);
                        if (otherTreeTO->treeSymbols.at(f) == FolderTreeSymbols::None) {
                            otherTreeTO->treeSymbols.at(f) = FolderTreeSymbols::Continue;
                        } else if (otherTreeTO->treeSymbols.at(f) == FolderTreeSymbols::End) {
                            otherTreeTO->treeSymbols.at(f) = FolderTreeSymbols::Branch;
                        } else {
                            break;
                        }
                    }

                }

            }

            //calc symbols before position numEqualFolders - 1
            for (int f = 0; f < numEqualFolders - 1; ++f) {
                if (prevTreeTO->treeSymbols.at(f) == FolderTreeSymbols::Branch) {
                    treeTO->treeSymbols.at(f) = FolderTreeSymbols::None;
                }
            }

            if (numEqualFolders < treeTO->folderNames.size()) {
                CHECK(numEqualFolders + 1 == treeTO->folderNames.size());
                treeTO->treeSymbols.back() = FolderTreeSymbols::Expanded;

                if (numEqualFolders > 0 && numEqualFolders < prevTreeTO->folderNames.size()) {
                    treeTO->treeSymbols.at(numEqualFolders - 1) = FolderTreeSymbols::End;
                    bool noneFound = false;
                    for (int j = i - 1; j >= 0; --j) {
                        auto& otherTreeTO = treeTOs.at(j);
                        if (otherTreeTO->treeSymbols.at(numEqualFolders - 1) == FolderTreeSymbols::None) {
                            otherTreeTO->treeSymbols.at(numEqualFolders - 1) = FolderTreeSymbols::Continue;
                            noneFound = true;
                        } else if (noneFound && otherTreeTO->treeSymbols.at(numEqualFolders - 1) == FolderTreeSymbols::End) {
                            otherTreeTO->treeSymbols.at(numEqualFolders - 1) = FolderTreeSymbols::Branch;
                        } else {
                            break;
                        }
                    }
                }
            }
            if (numEqualFolders > 0 && numEqualFolders < prevTreeTO->folderNames.size() && numEqualFolders == treeTO->folderNames.size()) {
                treeTO->treeSymbols.back() = FolderTreeSymbols::End;
            }
        }
    }

    //calc numLeafs and numReactions for folders
    for (int i = toInt(treeTOs.size()) - 1; i > 0; --i) {
        auto& treeTO = treeTOs.at(i);
        if (treeTO->isLeaf()) {
            int numReactions = 0;
            for (auto const& count : treeTO->getLeaf().rawTO->numLikesByEmojiType | std::views::values) {
                numReactions += count;
            }
            for (int j = i - 1; j >= 0; --j) {
                auto& otherTreeTO = treeTOs.at(j);
                auto numEqualFolders = getNumEqualFolders(treeTO->folderNames, otherTreeTO->folderNames);
                if (numEqualFolders == 0) {
                    break;
                }
                if (numEqualFolders == otherTreeTO->folderNames.size() && !otherTreeTO->isLeaf()) {
                    auto& folder = otherTreeTO->getFolder();
                    ++folder.numLeafs;
                    folder.numReactions += numReactions;
                }
            }
        }
    }

    //collapse items
    std::unordered_set<std::string> collapsedFolderStrings;
    for(auto const& folderNames : collapsedFolderNames) {
        collapsedFolderStrings.insert(boost::join(folderNames, "/"));
    }

    std::vector<NetworkResourceTreeTO> result;
    result.reserve(treeTOs.size());
    for (auto const& treeTO : treeTOs) {
        auto isVisible = true;

        std::string folderString;
        auto numSolderToCheck = treeTO->isLeaf() ? treeTO->folderNames.size() : treeTO->folderNames.size() - 1;
        for (size_t i = 0; i < numSolderToCheck; ++i) {
            folderString.append(treeTO->folderNames.at(i));
            if (collapsedFolderStrings.contains(folderString)) {
                isVisible = false;
            }
            if (i < numSolderToCheck) {
                folderString.append("/");
            }
        }

        if (!treeTO->isLeaf()) {
            auto folderString = boost::join(treeTO->folderNames, "/");
            if (collapsedFolderStrings.contains(folderString)) {
                treeTO->treeSymbols.back() = FolderTreeSymbols::Collapsed;
            }
        }
        if (isVisible) {
            result.emplace_back(treeTO);
        }
    }
    return result;
}

std::vector<NetworkResourceRawTO> NetworkResourceService::getMatchingRawTOs(NetworkResourceTreeTO const& treeTO, std::vector<NetworkResourceRawTO> const& rawTOs)
{
    if (treeTO->isLeaf()) {
        return {treeTO->getLeaf().rawTO};
    } else {
        auto findResult = _treeTOtoRawTOcache.find(treeTO);
        if (findResult != _treeTOtoRawTOcache.end()) {
            return findResult->second;
        }
        std::vector<NetworkResourceRawTO> result;
        for (auto const& [index, rawTO] : rawTOs | boost::adaptors::indexed(0)) {
            auto folderNames = getFolderNames(rawTO->resourceName);
            if (contains(folderNames, treeTO->folderNames)) {
                result.emplace_back(rawTO);
            }
        }
        _treeTOtoRawTOcache.emplace(treeTO, result);
        return result;
    }
}

void NetworkResourceService::invalidateCache()
{
    _treeTOtoRawTOcache.clear();
}

std::vector<std::string> NetworkResourceService::getFolderNames(std::string const& resourceName)
{
    std::vector<std::string> result;
    boost::split(result, resourceName, boost::is_any_of("/"));
    result.pop_back();
    return result;
}

std::set<std::vector<std::string>> NetworkResourceService::getFolderNames(std::vector<NetworkResourceRawTO> const& rawTOs, int minNesting)
{
    std::set<std::vector<std::string>> result;
    for (auto const& rawTO : rawTOs) {
        std::vector<std::string> folderNames;
        boost::split(folderNames, rawTO->resourceName, boost::is_any_of("/"));
        for (int i = 0; i < toInt(folderNames.size()) - minNesting; ++i) {
            result.insert(std::vector(folderNames.begin(), folderNames.begin() + minNesting + i));
        }
    }
    return result;
}

std::string NetworkResourceService::concatenateFolderName(std::vector<std::string> const& folderNames, bool withSlashAtTheEnd)
{
    auto result = boost::join(folderNames, "/");
    if (withSlashAtTheEnd) {
        result.append("/");
    }
    return result;
}

std::string NetworkResourceService::convertFolderNamesToSettings(std::set<std::vector<std::string>> const& data)
{
    std::vector<std::string> parts;
    for (auto const& folderNames : data) {
        parts.emplace_back(concatenateFolderName(folderNames, false));
    }
    return boost::join(parts, "\\");
}

std::set<std::vector<std::string>> NetworkResourceService::convertSettingsToFolderNames(std::string const& data)
{
    std::vector<std::string> parts;
    boost::split(parts, data, boost::is_any_of("\\"));

    std::set<std::vector<std::string>> result;
    for (auto const& part : parts) {
        std::vector<std::string> splittedParts;
        boost::split(splittedParts, part, boost::is_any_of("/"));
        result.insert(splittedParts);
    }
    return result;
}
