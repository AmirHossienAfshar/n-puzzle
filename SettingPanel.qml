import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15
import Pyside_Settings 1.0

ApplicationWindow {
    visible: true
    width: 400
    height: 350
    title: "Settings Panel"

    Pyside_Settings_class {
        id: settings
    }

    ColumnLayout {
        anchors.fill: parent
        spacing: 10

        // Puzzle Settings GroupBox
        GroupBox {
            id: puzzleSettings
            title: "Puzzle Settings"
            Layout.fillWidth: true
            Layout.fillHeight: true
            font.pixelSize: 14

            ColumnLayout {
                anchors.fill: parent
                spacing: 10

                GridLayout {
                    // anchors.horizontalCenter: parent.horizontalCenter
                    columns: 2
                    rowSpacing: 5
                    columnSpacing: 10

                    Label {
                        text: "Puzzle grid size:"
                    }
                    ComboBox {
                        Layout.fillWidth: true
                        model: ["3x3", "4x4", "5x5"]
                    }
                }
                Button {
                    text: "Generate Puzzle!"
                    Layout.fillWidth: true
                    Layout.alignment: Qt.AlignHCenter
                }
            }
        }

        GroupBox {
            id: agentSettings
            title: "Agent Settings"
            Layout.fillWidth: true
            Layout.fillHeight: true
            font.pixelSize: 14

            ColumnLayout {
                anchors.fill: parent
                spacing: 10

                GridLayout {
                    // anchors.horizontalCenter: parent.horizontalCenter
                    columns: 2
                    rowSpacing: 5
                    columnSpacing: 10

                    Label {
                        text: "Agent type:"
                    }
                    ComboBox {
                        Layout.fillWidth: true
                        model: ["Q-Learning", "A*", "Dijkstra"]
                    }
                    Label {
                        text: "Solver speed (step/sec):"
                    }
                    ComboBox {
                        Layout.fillWidth: true
                        model: ["1", "2", "4"]
                    }
                }
                Button {
                    text: "Train!"
                    Layout.fillWidth: true
                    Layout.alignment: Qt.AlignHCenter
                }
            }
        }
        GroupBox {
            id: trainingProgress
            title: "Training Progress"
            Layout.fillWidth: true
            Layout.fillHeight: false
            font.pixelSize: 14

            RowLayout {
                anchors.fill: parent
                spacing: 10

                Label {
                    text: "Training:"
                }
                ProgressBar {
                    Layout.fillWidth: true
                    value: 0.4  // Example progress value
                }
            }
        }
    }
}
