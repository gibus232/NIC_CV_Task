cmake_minimum_required(VERSION 3.21)

project(FamilyBudget
    VERSION 1.0.0
    DESCRIPTION "Family Budget Control App with Frutiger Aero Style"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

find_package(Qt6 6.4 REQUIRED COMPONENTS
    Core
    Quick
    QuickControls2
    Sql
    Charts
    Svg
)

qt_standard_project_setup()

qt_add_executable(FamilyBudget
    src/main.cpp
    src/database/database.h
    src/database/database.cpp
    src/models/usermanager.h
    src/models/usermanager.cpp
    src/models/budgetmanager.h
    src/models/budgetmanager.cpp
)

qt_add_qml_module(FamilyBudget
    URI FamilyBudget
    VERSION 1.0
    QML_FILES
        qml/main.qml
        qml/screens/LoginScreen.qml
        qml/screens/DashboardScreen.qml
        qml/screens/TransactionsScreen.qml
        qml/screens/AddTransactionScreen.qml
        qml/screens/ReportsScreen.qml
        qml/screens/SettingsScreen.qml
        qml/screens/FamilyScreen.qml
        qml/screens/BudgetScreen.qml
        qml/components/GlassPanel.qml
        qml/components/GlossyButton.qml
        qml/components/AeroHeader.qml
        qml/components/TransactionItem.qml
        qml/components/CategoryBadge.qml
        qml/components/UserAvatar.qml
        qml/components/BudgetBar.qml
        qml/components/BottomNav.qml
        qml/components/AeroDialog.qml
        qml/components/PinPad.qml
        qml/components/AmountDisplay.qml
    RESOURCES
        resources/fonts/placeholder.txt
)

set_target_properties(FamilyBudget PROPERTIES
    QT_ANDROID_PACKAGE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/android"
    QT_ANDROID_MIN_SDK_VERSION 24
    QT_ANDROID_TARGET_SDK_VERSION 34
    QT_ANDROID_VERSION_NAME "1.0.0"
    QT_ANDROID_VERSION_CODE 1
)

target_include_directories(FamilyBudget PRIVATE src)

target_link_libraries(FamilyBudget PRIVATE
    Qt6::Core
    Qt6::Quick
    Qt6::QuickControls2
    Qt6::Sql
    Qt6::Charts
    Qt6::Svg
)
