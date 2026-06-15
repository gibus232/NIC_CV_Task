import QtQuick
import QtQuick.Controls
import "../components"

Item {
    id: root
    property string month: budgetManager.currentMonth()

    function budgets() { return budgetManager.getBudgets(userManager.currentUserId, month) }
    function cats()    { return budgetManager.getCategories(userManager.currentUserId, true) }

    AeroHeader {
        id: hdr
        anchors { top: parent.top; left: parent.left; right: parent.right }
        title: "Бюджет"
        showAction: true; actionIcon: "+"
        onActionClicked: setBudgetPopup.open()
    }

    // Month selector
    Row {
        id: monthRow
        anchors { top: hdr.bottom; topMargin: 12; horizontalCenter: parent.horizontalCenter }; spacing: 12
        Rectangle {
            width: 36; height: 36; radius: 18; color: Qt.rgba(1,1,1,0.10); border.color: Qt.rgba(1,1,1,0.20); border.width: 1
            Text { anchors.centerIn: parent; text: "‹"; color: "white"; font.pixelSize: 24 }
            MouseArea { anchors.fill: parent; onClicked: { root.month = budgetManager.prevMonth(root.month); budgetList.model = root.budgets() } }
        }
        Text { anchors.verticalCenter: parent.verticalCenter; text: budgetManager.monthDisplayName(root.month); color: "white"; font.pixelSize: 16; font.weight: Font.Medium }
        Rectangle {
            width: 36; height: 36; radius: 18; color: Qt.rgba(1,1,1,0.10); border.color: Qt.rgba(1,1,1,0.20); border.width: 1
            Text { anchors.centerIn: parent; text: "›"; color: "white"; font.pixelSize: 24 }
            MouseArea { anchors.fill: parent; onClicked: { root.month = budgetManager.nextMonth(root.month); budgetList.model = root.budgets() } }
        }
    }

    ListView {
        id: budgetList
        anchors { top: monthRow.bottom; topMargin: 12; bottom: parent.bottom; bottomMargin: 8; left: parent.left; leftMargin: 16; right: parent.right; rightMargin: 16 }
        spacing: 10; clip: true
        model: root.budgets()

        delegate: BudgetBar {
            width: budgetList.width
            label: modelData.name
            barColor: modelData.color || "#1565C0"
            budget: modelData.budget || 0
            spent:  modelData.spent  || 0
        }

        Text {
            visible: budgetList.count === 0
            anchors.centerIn: parent
            text: "Бюджеты не заданы.\nНажмите + чтобы добавить."
            color: Qt.rgba(1,1,1,0.40); font.pixelSize: 15; horizontalAlignment: Text.AlignHCenter
        }
    }

    Connections {
        target: budgetManager
        function onDataChanged() { budgetList.model = root.budgets() }
    }

    // Set budget popup
    Popup {
        id: setBudgetPopup
        anchors.centerIn: parent
        width: Math.min(parent.width - 40, 340)
        modal: true; dim: true

        property int selectedCatId: -1
        property string selectedCatName: ""

        onOpened: { amtIn.text = "0"; setBudgetPopup.selectedCatId = -1 }

        background: Rectangle {
            radius: 24; color: Qt.rgba(0.05,0.10,0.25,0.96)
            border.color: Qt.rgba(1,1,1,0.30); border.width: 1
            Rectangle { anchors { top: parent.top; left: parent.left; right: parent.right; margins: 1 }; height: 48; radius: parent.radius; gradient: Gradient { GradientStop { position: 0.0; color: Qt.rgba(1,1,1,0.12) }; GradientStop { position: 1.0; color: Qt.rgba(1,1,1,0.00) } } }
        }
        Overlay.modal: Rectangle { color: Qt.rgba(0,0,0,0.60) }

        Column {
            width: parent.width; spacing: 14; padding: 24

            Text { anchors.horizontalCenter: parent.horizontalCenter; text: "Установить бюджет"; color: "white"; font.pixelSize: 18; font.weight: Font.Medium }

            Text { text: "Категория:"; color: Qt.rgba(1,1,1,0.65); font.pixelSize: 13 }

            Flow {
                width: parent.width - 48; anchors.horizontalCenter: parent.horizontalCenter; spacing: 8
                Repeater {
                    model: root.cats()
                    delegate: CategoryBadge {
                        label: modelData.name
                        badgeColor: modelData.color || "#1565C0"
                        selected: setBudgetPopup.selectedCatId === modelData.id
                        onClicked: { setBudgetPopup.selectedCatId = modelData.id; setBudgetPopup.selectedCatName = modelData.name }
                    }
                }
            }

            Text { text: "Сумма бюджета:"; color: Qt.rgba(1,1,1,0.65); font.pixelSize: 13 }

            Rectangle {
                width: parent.width - 48; height: 46; radius: 12; anchors.horizontalCenter: parent.horizontalCenter
                color: Qt.rgba(1,1,1,0.10); border.color: Qt.rgba(1,1,1,0.25); border.width: 1
                TextInput { id: amtIn; anchors { fill: parent; leftMargin: 14; rightMargin: 14 }; verticalAlignment: TextInput.AlignVCenter; color: "white"; font.pixelSize: 18; text: "0"; inputMethodHints: Qt.ImhFormattedNumbersOnly }
            }

            Row {
                anchors.horizontalCenter: parent.horizontalCenter; spacing: 12
                Rectangle { width: 100; height: 44; radius: 22; color: Qt.rgba(1,1,1,0.10); border.color: Qt.rgba(1,1,1,0.20); border.width: 1; Text { anchors.centerIn: parent; text: "Отмена"; color: "white"; font.pixelSize: 14 }; MouseArea { anchors.fill: parent; onClicked: setBudgetPopup.close() } }
                Rectangle {
                    width: 130; height: 44; radius: 22
                    gradient: Gradient { orientation: Gradient.Horizontal; GradientStop { position: 0.0; color: "#1E88E5" }; GradientStop { position: 1.0; color: "#1565C0" } }
                    border.color: Qt.rgba(1,1,1,0.35); border.width: 1
                    Text { anchors.centerIn: parent; text: "Сохранить"; color: "white"; font.pixelSize: 14; font.weight: Font.Medium }
                    MouseArea {
                        anchors.fill: parent
                        onClicked: {
                            if (setBudgetPopup.selectedCatId >= 0) {
                                budgetManager.setBudget(userManager.currentUserId, setBudgetPopup.selectedCatId, parseFloat(amtIn.text)||0, root.month)
                                setBudgetPopup.close()
                            }
                        }
                    }
                }
            }
        }
    }
}
