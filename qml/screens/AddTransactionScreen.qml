import QtQuick
import QtQuick.Controls
import "../components"

Item {
    id: root
    property bool isExpense: true
    property int selectedCategoryId: -1
    property double amount: 0
    property string note: ""
    property string date: Qt.formatDate(new Date(), "yyyy-MM-dd")

    property var categories: budgetManager.getCategories(userManager.currentUserId, root.isExpense)

    AeroHeader {
        id: hdr
        anchors { top: parent.top; left: parent.left; right: parent.right }
        title: root.isExpense ? "Расход" : "Доход"
        showBack: true
        onBackClicked: StackView.view.pop()
    }

    Flickable {
        anchors { top: hdr.bottom; left: parent.left; right: parent.right; bottom: saveBtn.top; bottomMargin: 12 }
        contentHeight: mainCol.height + 20; clip: true

        Column {
            id: mainCol
            width: parent.width; spacing: 20

            // Type toggle
            Rectangle {
                width: 220; height: 44; radius: 22
                anchors.horizontalCenter: parent.horizontalCenter
                color: Qt.rgba(1,1,1,0.08); border.color: Qt.rgba(1,1,1,0.20); border.width: 1

                Row {
                    anchors.fill: parent

                    Rectangle {
                        width: parent.width / 2; height: parent.height; radius: parent.parent.radius
                        color: root.isExpense ? "#C62828" : "transparent"
                        Behavior on color { ColorAnimation { duration: 200 } }
                        Text { anchors.centerIn: parent; text: "↓ Расход"; color: "white"; font.pixelSize: 14; font.weight: Font.Medium }
                        MouseArea { anchors.fill: parent; onClicked: { root.isExpense = true; root.selectedCategoryId = -1; root.categories = budgetManager.getCategories(userManager.currentUserId, true) } }
                    }
                    Rectangle {
                        width: parent.width / 2; height: parent.height; radius: parent.parent.radius
                        color: !root.isExpense ? "#1B5E20" : "transparent"
                        Behavior on color { ColorAnimation { duration: 200 } }
                        Text { anchors.centerIn: parent; text: "↑ Доход"; color: "white"; font.pixelSize: 14; font.weight: Font.Medium }
                        MouseArea { anchors.fill: parent; onClicked: { root.isExpense = false; root.selectedCategoryId = -1; root.categories = budgetManager.getCategories(userManager.currentUserId, false) } }
                    }
                }
            }

            // Amount input
            Column {
                anchors.horizontalCenter: parent.horizontalCenter; spacing: 4
                Text { anchors.horizontalCenter: parent.horizontalCenter; text: "Сумма"; color: Qt.rgba(1,1,1,0.60); font.pixelSize: 13 }

                Row {
                    anchors.horizontalCenter: parent.horizontalCenter; spacing: 4
                    Text { anchors.verticalCenter: parent.verticalCenter; text: root.isExpense ? "−" : "+"; color: root.isExpense ? "#EF5350" : "#66BB6A"; font.pixelSize: 32; font.weight: Font.Light }
                    TextInput {
                        id: amtInput
                        anchors.verticalCenter: parent.verticalCenter
                        text: "0"; color: root.isExpense ? "#EF5350" : "#66BB6A"
                        font.pixelSize: 48; font.weight: Font.Light
                        inputMethodHints: Qt.ImhFormattedNumbersOnly
                        validator: RegularExpressionValidator { regularExpression: /^\d{0,9}(\.\d{0,2})?$/ }
                        onTextChanged: root.amount = parseFloat(text) || 0
                        cursorVisible: activeFocus
                    }
                    Text { anchors.verticalCenter: parent.verticalCenter; text: " ₽"; color: Qt.rgba(1,1,1,0.50); font.pixelSize: 28 }
                }
            }

            // Category picker
            Column {
                width: parent.width - 32; anchors.horizontalCenter: parent.horizontalCenter; spacing: 10
                Text { text: "Категория"; color: Qt.rgba(1,1,1,0.65); font.pixelSize: 13 }

                Flow {
                    width: parent.width; spacing: 8
                    Repeater {
                        model: root.categories
                        delegate: CategoryBadge {
                            label: modelData.name
                            badgeColor: modelData.color || "#1565C0"
                            selected: root.selectedCategoryId === modelData.id
                            onClicked: root.selectedCategoryId = modelData.id
                        }
                    }
                }
            }

            // Date
            Column {
                width: parent.width - 32; anchors.horizontalCenter: parent.horizontalCenter; spacing: 6
                Text { text: "Дата"; color: Qt.rgba(1,1,1,0.65); font.pixelSize: 13 }
                Rectangle {
                    width: parent.width; height: 46; radius: 12
                    color: Qt.rgba(1,1,1,0.10); border.color: Qt.rgba(1,1,1,0.22); border.width: 1
                    TextInput {
                        id: dateIn
                        anchors { fill: parent; leftMargin: 14; rightMargin: 14 }
                        verticalAlignment: TextInput.AlignVCenter
                        text: root.date; color: "white"; font.pixelSize: 15
                        onTextChanged: root.date = text
                    }
                }
            }

            // Note
            Column {
                width: parent.width - 32; anchors.horizontalCenter: parent.horizontalCenter; spacing: 6
                Text { text: "Примечание"; color: Qt.rgba(1,1,1,0.65); font.pixelSize: 13 }
                Rectangle {
                    width: parent.width; height: 46; radius: 12
                    color: Qt.rgba(1,1,1,0.10); border.color: Qt.rgba(1,1,1,0.22); border.width: 1
                    TextInput {
                        anchors { fill: parent; leftMargin: 14; rightMargin: 14 }
                        verticalAlignment: TextInput.AlignVCenter
                        placeholderText: "Необязательно..."
                        placeholderTextColor: Qt.rgba(1,1,1,0.35)
                        color: "white"; font.pixelSize: 15
                        onTextChanged: root.note = text
                    }
                }
            }
        }
    }

    // Save button
    Rectangle {
        id: saveBtn
        anchors { bottom: parent.bottom; bottomMargin: 30; left: parent.left; leftMargin: 24; right: parent.right; rightMargin: 24 }
        height: 54; radius: 27
        opacity: root.amount > 0 && root.selectedCategoryId >= 0 ? 1.0 : 0.45
        gradient: Gradient {
            orientation: Gradient.Horizontal
            GradientStop { position: 0.0; color: root.isExpense ? "#E53935" : "#2E7D32" }
            GradientStop { position: 1.0; color: root.isExpense ? "#C62828" : "#1B5E20" }
        }
        border.color: Qt.rgba(1,1,1,0.35); border.width: 1

        Rectangle {
            anchors { top: parent.top; left: parent.left; right: parent.right; margins: 1 }
            height: parent.height * 0.50; radius: parent.radius
            gradient: Gradient {
                GradientStop { position: 0.0; color: Qt.rgba(1,1,1,0.22) }
                GradientStop { position: 1.0; color: Qt.rgba(1,1,1,0.00) }
            }
        }

        Text { anchors.centerIn: parent; text: "Сохранить"; color: "white"; font.pixelSize: 17; font.weight: Font.Medium }

        scale: saveMa.pressed ? 0.96 : 1.0
        Behavior on scale { NumberAnimation { duration: 80 } }

        MouseArea {
            id: saveMa; anchors.fill: parent
            enabled: root.amount > 0 && root.selectedCategoryId >= 0
            onClicked: {
                budgetManager.addTransaction(userManager.currentUserId, root.selectedCategoryId,
                                             root.amount, root.isExpense, root.note, root.date)
                StackView.view.pop()
            }
        }
    }
}
