import QtQuick 2.15
import QtQuick.Controls 2.15
import Pyside_Bridge 1.0
import QtQuick.Layouts 1.15

ApplicationWindow {
    visible: true
    width: 700
    height: 400
    title: "Sliding Puzzle (AI-Controlled)"

    Pyside_Bridge_class {
        id: bridge
        // pyside_invoke_generate_btn
        // pyside_invoke_start_btn
    }

    property int gridSize: bridge.pyside_puzzle_size
    property real tileSize: width / gridSize

    RowLayout {
        anchors.fill: parent

        GroupBox {
            title: "Puzzle Board"
            Layout.fillWidth: true
            Layout.fillHeight: true
            PuzzleBoard {
                id: board
                model: bridge.pyside_puzzle_list
                tileSize: width / bridge.pyside_puzzle_size
            }
        }
        
        SettingPanel {
            Layout.preferredWidth: 300
            Layout.fillHeight: true
            trainingProgressValue: bridge.pyside_training_progress
            startSolverEnabled: bridge.pyside_invoke_start_btn
            generatePuzzleEnabled: bridge.pyside_invoke_generate_btn
            searchPuzzleEnabled: bridge.pyside_search_btn_is_enable
            searchPendingLableStatus: bridge.pyside_search_status_is_pending
            searchDoneLableStatus: bridge.pyside_search_status_is_done
            seachProgressBusy: bridge.pyside_search_status_progress_is_busy
        }
    }
}
