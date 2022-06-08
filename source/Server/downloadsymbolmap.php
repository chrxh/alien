<?php
    require './helpers.php';

    $db = connectToDB();
    $db->begin_transaction();

    $id = $_GET["id"];

    if ($response = $db->query("SELECT sim.ID as id, sim.SYMBOL_MAP as symbolMap FROM simulation sim where ID=$id")) {
        $obj = $response->fetch_object();
        echo $obj->symbolMap;
    }
    else {
        echo json_encode(["result"=>false]);
    }

    $db->commit();
    $db->close();
?>