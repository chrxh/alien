#include "BrowserWindow.h"

#include <imgui.h>

#define CPPHTTPLIB_OPENSSL_SUPPORT
#include <httplib.h>

#include "AlienImGui.h"
#include "GlobalSettings.h"
#include "StyleRepository.h"

_BrowserWindow::_BrowserWindow(SimulationController const& simController)
    : _AlienWindow("Browser", "browser.network", false)
    , _simController(simController)
{
    if (_on) {
        processActivated();
    }
}

_BrowserWindow::~_BrowserWindow()
{
}

enum MyItemColumnID
{
    MyItemColumnID_ID,
    MyItemColumnID_Name,
    MyItemColumnID_Action,
    MyItemColumnID_Quantity,
    MyItemColumnID_Description
};

struct MyItem
{
    int ID;
    const char* Name;
    int Quantity;

    // We have a problem which is affecting _only this demo_ and should not affect your code:
    // As we don't rely on std:: or other third-party library to compile dear imgui, we only have reliable access to qsort(),
    // however qsort doesn't allow passing user data to comparing function.
    // As a workaround, we are storing the sort specs in a static/global for the comparing function to access.
    // In your own use case you would probably pass the sort specs to your sorting/comparing functions directly and not use a global.
    // We could technically call ImGui::TableGetSortSpecs() in CompareWithSortSpecs(), but considering that this function is called
    // very often by the sorting algorithm it would be a little wasteful.
    static const ImGuiTableSortSpecs* s_current_sort_specs;

    // Compare function to be used by qsort()
    static int CompareWithSortSpecs(const void* lhs, const void* rhs)
    {
        const MyItem* a = (const MyItem*)lhs;
        const MyItem* b = (const MyItem*)rhs;
        for (int n = 0; n < s_current_sort_specs->SpecsCount; n++) {
            // Here we identify columns using the ColumnUserID value that we ourselves passed to TableSetupColumn()
            // We could also choose to identify columns based on their index (sort_spec->ColumnIndex), which is simpler!
            const ImGuiTableColumnSortSpecs* sort_spec = &s_current_sort_specs->Specs[n];
            int delta = 0;
            switch (sort_spec->ColumnUserID) {
            case MyItemColumnID_ID:
                delta = (a->ID - b->ID);
                break;
            case MyItemColumnID_Name:
                delta = (strcmp(a->Name, b->Name));
                break;
            case MyItemColumnID_Quantity:
                delta = (a->Quantity - b->Quantity);
                break;
            case MyItemColumnID_Description:
                delta = (strcmp(a->Name, b->Name));
                break;
            default:
                IM_ASSERT(0);
                break;
            }
            if (delta > 0)
                return (sort_spec->SortDirection == ImGuiSortDirection_Ascending) ? +1 : -1;
            if (delta < 0)
                return (sort_spec->SortDirection == ImGuiSortDirection_Ascending) ? -1 : +1;
        }

        // qsort() is instable so always return a way to differenciate items.
        // Your own compare function may want to avoid fallback on implicit sort specs e.g. a Name compare if it wasn't already part of the sort specs.
        return (a->ID - b->ID);
    }
};
const ImGuiTableSortSpecs* MyItem::s_current_sort_specs = NULL;

void _BrowserWindow::processIntern()
{
    // Create item list
    static const char* template_items_names[] = {
        "Banana",
        "Apple",
        "Cherry",
        "Watermelon",
        "Grapefruit",
        "Strawberry",
        "Mango",
        "Kiwi",
        "Orange",
        "Pineapple",
        "Blueberry",
        "Plum",
        "Coconut",
        "Pear",
        "Apricot"};

    static ImVector<MyItem> items;
    if (items.Size == 0) {
        items.resize(50, MyItem());
        for (int n = 0; n < items.Size; n++) {
            const int template_n = n % IM_ARRAYSIZE(template_items_names);
            MyItem& item = items[n];
            item.ID = n;
            item.Name = template_items_names[template_n];
            item.Quantity = (n * n - n) % 20;  // Assign default quantities
        }
    }

    static ImGuiTableFlags flags = ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable | ImGuiTableFlags_Sortable
        | ImGuiTableFlags_SortMulti | ImGuiTableFlags_RowBg | ImGuiTableFlags_BordersOuter | ImGuiTableFlags_BordersV | ImGuiTableFlags_NoBordersInBody
        | ImGuiTableFlags_ScrollY;
    if (ImGui::BeginTable("table_sorting", 4, flags, ImVec2(0.0f, 0.0f), 0.0f)) {
        // Declare columns
        // We use the "user_id" parameter of TableSetupColumn() to specify a user id that will be stored in the sort specifications.
        // This is so our sort function can identify a column given our own identifier. We could also identify them based on their index!
        // Demonstrate using a mixture of flags among available sort-related flags:
        // - ImGuiTableColumnFlags_DefaultSort
        // - ImGuiTableColumnFlags_NoSort / ImGuiTableColumnFlags_NoSortAscending / ImGuiTableColumnFlags_NoSortDescending
        // - ImGuiTableColumnFlags_PreferSortAscending / ImGuiTableColumnFlags_PreferSortDescending
        ImGui::TableSetupColumn("ID", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthFixed, 0.0f, MyItemColumnID_ID);
        ImGui::TableSetupColumn("Name", ImGuiTableColumnFlags_WidthFixed, 0.0f, MyItemColumnID_Name);
        ImGui::TableSetupColumn("Action", ImGuiTableColumnFlags_NoSort | ImGuiTableColumnFlags_WidthFixed, 0.0f, MyItemColumnID_Action);
        ImGui::TableSetupColumn("Quantity", ImGuiTableColumnFlags_PreferSortDescending | ImGuiTableColumnFlags_WidthStretch, 0.0f, MyItemColumnID_Quantity);
        ImGui::TableSetupScrollFreeze(0, 1);  // Make row always visible
        ImGui::TableHeadersRow();

        // Sort our data if sort specs have been changed!
        if (ImGuiTableSortSpecs* sorts_specs = ImGui::TableGetSortSpecs())
            if (sorts_specs->SpecsDirty) {
                MyItem::s_current_sort_specs = sorts_specs;  // Store in variable accessible by the sort function.
                if (items.Size > 1)
                    qsort(&items[0], (size_t)items.Size, sizeof(items[0]), MyItem::CompareWithSortSpecs);
                MyItem::s_current_sort_specs = NULL;
                sorts_specs->SpecsDirty = false;
            }

        // Demonstrate using clipper for large vertical lists
        ImGuiListClipper clipper;
        clipper.Begin(items.Size);
        while (clipper.Step())
            for (int row_n = clipper.DisplayStart; row_n < clipper.DisplayEnd; row_n++) {
                // Display a data item
                MyItem* item = &items[row_n];
                ImGui::PushID(item->ID);
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("%04d", item->ID);
                ImGui::TableNextColumn();
                ImGui::TextUnformatted(item->Name);
                ImGui::TableNextColumn();
                ImGui::SmallButton("None");
                ImGui::TableNextColumn();
                ImGui::Text("%d", item->Quantity);
                ImGui::PopID();
            }
        ImGui::EndTable();
    }
}

void _BrowserWindow::processActivated()
{
    _server = GlobalSettings::getInstance().getStringState("settings.server", "alien-project.org");

    httplib::SSLClient cli(_server);
    cli.set_ca_cert_path("./resources/ca-bundle.crt");
    cli.enable_server_certificate_verification(true);

    auto result = cli.get_openssl_verify_result();
/*
    if (result) {
        std::cout << "verify error: " << X509_verify_cert_error_string(result) << std::endl;
    }
*/

    if (auto res = cli.Get("/world-explorer/api/getsimulationinfos.php")) {
        _test = res->body;
    }
}
